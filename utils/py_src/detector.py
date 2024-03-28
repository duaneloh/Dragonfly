'''Module containing detector class'''

import sys
import os
import numpy as np
from numpy import ma
import pandas
try:
    import h5py
    HDF5_MODE = True
except ImportError:
    HDF5_MODE = False

class Detector(object):
    """Dragonfly detector

    The detector file format is specified in github.com/duaneloh/Dragonfly/wiki
    This class reads the file and provides numpy arrays which can be used for
    further processing.

    __init__ arguments (optional):
        det_fname (string) - Path to detector file to populate attributes
        detd_pix (float) - Detector distance in pixels (detd/pixsize)
        ewald_rad (float) - Ewald sphere radius in voxels. If in doubt, = detd_pix
        mask_flag (bool) - Whether to read the mask column for each pixel
        keep_mask_1 (bool) - Whether to consider mask=1 pixels as good

    For the new ASCII format, detd_pix and ewald_rad numbers are read from the file \
    but for the old file, they must be provided.

    Methods:
        parse(fname, mask_flag=False, keep_mask_1=True)
        write(fname)
        assemble_frame(data, zoomed=False, sym=False)
        calc_from_coords()

    On parsing, it produces the following numpy arrays (each of length num_pix)

    Attributes:
        self.qx, self.qy, self.qz - Voxel space coordinates (origin at (0,0,0))
        self.cx, self.cy - Floating point 2D coordinates (origin at (0,0))
        self.x, self.y - Integer and shifted 2D coordinates (corner at (0,0))
        self.mask - Assembled mask
        self.raw_mask - Unassembled mask as stored in detector file
        self.unassembled_mask - Unassembled mask (1=good, 0=bad)
    """
    def __init__(self, det_fname=None, detd_pix=None,
                 ewald_rad=None, mask_flag=False, keep_mask_1=True):
        self.detd = detd_pix
        self.ewald_rad = ewald_rad
        self.background = None
        self._sym_shape = None
        if det_fname is not None:
            self.parse(det_fname, mask_flag, keep_mask_1)

    def parse(self, fname, mask_flag=False, keep_mask_1=True):
        """ Parse Dragonfly detector from file

        File can either be in the HDF5 or ASCII format
        """
        self.det_fname = fname
        if HDF5_MODE and h5py.is_hdf5(self.det_fname):
            self._parse_h5det(mask_flag, keep_mask_1)
        elif os.path.splitext(self.det_fname)[1] == '.h5':
            fheader = np.fromfile(self.det_fname, '=c', count=8)
            if fheader == chr(137)+'HDF\r\n'+chr(26)+'\n':
                if not HDF5_MODE:
                    raise IOError('Unable to parse HDF5 detector')
                else:
                    self._parse_h5det(mask_flag, keep_mask_1)
            else:
                self._parse_asciidet(mask_flag, keep_mask_1)
        else:
            self._parse_asciidet(mask_flag, keep_mask_1)

    def write(self, fname):
        """ Write Dragonfly detector to file

        If h5py is available and the file name as a '.h5' extension,
        an HDF5 detector will be written, otherwise an ASCII file will be generated.

        Note that the background array can only be stored in an HDF5 detector
        """
        try:
            val = self.qx + self.qy + self.qz + self.corr + self.raw_mask
            val = self.detd + self.ewald_rad
        except AttributeError:
            print('Detector attributes not populated. Cannot write to file')
            print('Need qx, qy, qz, corr, raw_mask, detd and ewald_rad')
            return

        if os.path.splitext(fname)[1] == '.h5':
            if HDF5_MODE:
                self._write_h5det(fname)
            else:
                raise IOError('Unable to write HDF5 detector without h5py')
        else:
            print('Writing ASCII detector file')
            self._write_asciidet(fname)

    def assemble_frame(self, data, zoomed=False, sym=False, avg=False):
        ''' Assemble given raw image

        Arguments:
            data - array of num_pix values
            zoomed (bool) - Restrict assembled image to non-masked pixels
            sym (bool) - Centro-symmetrize image

        Returns:
            Numpy masked array representing assembled image
        '''
        if sym:
            self._init_sym()
            img = ma.masked_array(np.zeros(self._sym_shape, dtype='f8'), mask=1-self._sym_mask)
            np.add.at(img, (self._sym_x, self._sym_y), data*self.unassembled_mask)
            np.add.at(img, (self._sym_fx, self._sym_fy), data*self.unassembled_mask)

            if avg:
                countimg = np.zeros(self._sym_shape, dtype='f8')
                np.add.at(countimg, (self._sym_x, self._sym_y), self.unassembled_mask)
                np.add.at(countimg, (self._sym_fx, self._sym_fy), self.unassembled_mask)
                img.data[countimg>0] /= countimg[countimg>0]
            else:
                img.data[self._sym_bothgood] /= 2.

            if zoomed:
                b = self._sym_zoom_bounds
                return img[b[0]:b[1], b[2]:b[3]]
        else:
            img = ma.masked_array(np.zeros(self.frame_shape, dtype='f8'), mask=1-self.mask)
            np.add.at(img, (self.x, self.y), data*self.unassembled_mask)
            if avg:
                countimg = np.zeros(self.frame_shape, dtype='f8')
                np.add.at(countimg, (self.x, self.y), self.unassembled_mask)
                img.data[countimg>0] /= countimg[countimg>0]
            if zoomed:
                b = self.zoom_bounds
                return img[b[0]:b[1], b[2]:b[3]]
        return img

    def calc_from_coords(self):
        ''' Calculate essential detector attributes from pixel coordinates

        Needs:
            cx, cy, detd, ewald_rad
        Calculates:
            qx, qy, qz and corr
        '''
        try:
            val = self.cx + self.cy
            val = self.detd + self.ewald_rad
        except AttributeError:
            print('Need cx, cy, detd and ewald_rad to be defined')
            print('detd must have same units as cx and cy')
            print('ewald_rad should be in voxel units')
            return

        fac = np.sqrt(self.cx**2 + self.cy**2 + self.detd**2)
        self.qx = self.cx * self.ewald_rad / fac
        self.qy = self.cy * self.ewald_rad / fac
        self.qz = self.ewald_rad * (self.detd/fac - 1.)
        self.corr = self.detd / fac**3 * (1. - self.cx**2 / fac**2)

    def _parse_asciidet(self, mask_flag, keep_mask_1):
        """ (Internal) Detector file parser

        Arguments:
            mask_flag (bool, optional) - Whether to read the mask column
            keep_mask_1 (bool, optional) - Whether to keep mask=1 within the boolean mask
        """
        print('Parsing ASCII detector file')
        self._check_header()
        sys.stderr.write('Reading %s...'%self.det_fname)
        if mask_flag:
            sys.stderr.write('with mask...')
        dframe = pandas.read_csv(
            self.det_fname,
            delim_whitespace=True, skiprows=1, engine='c', header=None,
            names=['qx', 'qy', 'qz', 'corr', 'mask'],
            dtype={'qx':'f8', 'qy':'f8', 'qz':'f8', 'corr':'f8', 'mask':'u1'})
        self.qx, self.qy, self.qz, self.corr = tuple([np.array(dframe[key]) # pylint: disable=C0103
                                                      for key in ['qx', 'qy', 'qz', 'corr']])
        self.raw_mask = np.array(dframe['mask']).astype('u1')
        sys.stderr.write('done\n')
        self._process_det(mask_flag, keep_mask_1)

    def _parse_h5det(self, mask_flag, keep_mask_1):
        print('Parsing HDF5 detector file')
        sys.stderr.write('Reading %s...'%self.det_fname)
        if mask_flag:
            sys.stderr.write('with mask...')
        with h5py.File(self.det_fname, 'r') as fptr:
            self.qx = fptr['qx'][:]
            self.qy = fptr['qy'][:]
            self.qz = fptr['qz'][:]
            self.corr = fptr['corr'][:]
            self.raw_mask = fptr['mask'][:].astype('u1')
            self.detd = fptr['detd'][()]
            self.ewald_rad = fptr['ewald_rad'][()]
            if 'background' in fptr:
                self.background = fptr['background'][:]
        sys.stderr.write('done\n')
        self._process_det(mask_flag, keep_mask_1)

    def _write_asciidet(self, fname):
        print('Writing ASCII detector file')
        qx = self.qx.ravel()
        qy = self.qy.ravel()
        qz = self.qz.ravel()
        corr = self.corr.ravel()
        mask = self.raw_mask.ravel().astype('u1')

        with open(fname, "w") as fptr:
            fptr.write("%d %.6f %.6f\n" % (qx.size, self.detd, self.ewald_rad))
            for par0, par1, par2, par3, par4 in zip(qx, qy, qz, corr, mask):
                txt = "%21.15e %21.15e %21.15e %21.15e %d\n" % (par0, par1, par2, par3, par4)
                fptr.write(txt)

    def _write_h5det(self, fname):
        print('Writing HDF5 detector file')
        with h5py.File(fname, "w") as fptr:
            fptr['qx'] = self.qx.ravel().astype('f8')
            fptr['qy'] = self.qy.ravel().astype('f8')
            fptr['qz'] = self.qz.ravel().astype('f8')
            fptr['corr'] = self.corr.ravel().astype('f8')
            fptr['mask'] = self.raw_mask.ravel().astype('u1')
            fptr['detd'] = float(self.detd)
            fptr['ewald_rad'] = float(self.ewald_rad)
            if self.background is not None:
                fptr['background'] = self.background.ravel().astype('f8')

    def _check_header(self):
        with open(self.det_fname, 'r') as fptr:
            line = fptr.readline().rstrip().split()
        if len(line) > 1:
            self.detd = float(line[1])
            self.ewald_rad = float(line[2])
        else:
            if self.detd is None:
                raise TypeError('Old type detector file. Need detd_pix')
            if self.ewald_rad is None:
                raise TypeError('Old type detector file. Need ewald_rad')

    def _process_det(self, mask_flag, keep_mask_1):
        if mask_flag:
            mask = np.copy(self.raw_mask)
            if keep_mask_1:
                mask[mask == 1] = 0 # To keep both 0 and 1
                mask = mask // 2 # To keep both 0 and 1
            else:
                mask[mask == 2] = 1 # To keep only mask==0
            mask = 1 - mask
        else:
            self.raw_mask = np.zeros(self.qx.shape, dtype='u1')
            mask = np.ones(self.qx.shape, dtype='u1')

        if self.qz.mean() > 0:
            self.cx = self.qx * self.detd / (self.ewald_rad - self.qz) # pylint: disable=C0103
            self.cy = self.qy * self.detd / (self.ewald_rad - self.qz) # pylint: disable=C0103
        else:
            self.cx = self.qx * self.detd / (self.ewald_rad + self.qz) # pylint: disable=C0103
            self.cy = self.qy * self.detd / (self.ewald_rad + self.qz) # pylint: disable=C0103
        self.x = np.round(self.cx - self.cx.min()).astype('i4')
        self.y = np.round(self.cy - self.cy.min()).astype('i4')
        self.unassembled_mask = mask.ravel()
        self._init_assem()

    def _init_assem(self):
        # Calculate attributes given self.x and self.y
        mask = self.unassembled_mask
        self.frame_shape = (self.x.max()+1, self.y.max()+1)

        self.mask = np.zeros(self.frame_shape, dtype='u1')
        self.mask[self.x, self.y] = mask
        self.mask = np.sign(self.mask)

        xsel = self.x[mask.astype('bool')]
        ysel = self.y[mask.astype('bool')]
        self.zoom_bounds = (xsel.min(), xsel.max()+1, ysel.min(), ysel.max()+1)

    def _init_sym(self, force=False):
        if self._sym_shape is not None and not force:
            return
        self._sym_shape = (2*int(np.ceil(np.abs(self.cx).max()))+1,
                           2*int(np.ceil(np.abs(self.cy).max()))+1)

        self._sym_x = np.round(self.cx + self._sym_shape[0]//2).astype('i4')
        self._sym_y = np.round(self.cy + self._sym_shape[1]//2).astype('i4')
        self._sym_fx = self._sym_shape[0] - 1 - self._sym_x
        self._sym_fy = self._sym_shape[1] - 1 - self._sym_y

        self._sym_mask = np.zeros(self._sym_shape, dtype='u1')
        np.add.at(self._sym_mask, (self._sym_x, self._sym_y), self.unassembled_mask)
        np.add.at(self._sym_mask, (self._sym_fx, self._sym_fy), self.unassembled_mask)
        self._sym_bothgood = (self._sym_mask == 2)
        self._sym_mask = np.sign(self._sym_mask)

        mask = self.unassembled_mask
        xsel = np.concatenate((self._sym_x[mask.astype('bool')], self._sym_fx[mask.astype('bool')]))
        ysel = np.concatenate((self._sym_y[mask.astype('bool')], self._sym_fy[mask.astype('bool')]))
        self._sym_zoom_bounds = (xsel.min(), xsel.max()+1, ysel.min(), ysel.max()+1)

    @property
    def coords_xy(self):
        '''Return 2D pixel coordinates'''
        return self.cx, self.cy

    @property
    def qvals_xyz(self):
        '''Return 3D voxel values'''
        return self.qx, self.qy, self.qz

    @property
    def indices_xy(self):
        '''Return 2D integer coordinates (for assembly)
        Corner of the detector at (0,0)'''
        return self.x, self.y
