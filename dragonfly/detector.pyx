'''Module containing detector class'''

import sys
import os
import numpy as np
from numpy import ma
import pandas
import h5py

from . cimport detector as c_det
from .detector cimport CDetector
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcpy
from libc.math cimport sqrt

cdef class CDetector:
    """Dragonfly detector cython class

    The detector file format is specified in github.com/duaneloh/Dragonfly/wiki
    This class reads the file and provides numpy arrays which can be used for
    further processing.

    __init__ arguments (optional):
        fname (string) - Path to detector file to populate attributes
        mask_flag (bool) - Whether to read the mask column for each pixel
        keep_mask_1 (bool) - Whether to consider mask=1 pixels as good

    On parsing, it produces the following numpy arrays (each of length num_pix)

    Attributes:
        self.qvals - Voxel space coordinates (origin at (0,0,0))
        self.cx, self.cy - Floating point 2D coordinates (origin at (0,0))
        self.x, self.y - Integer and shifted 2D coordinates (corner at (0,0))
        self.raw_mask - Unassembled mask as stored in detector file
        self.mask - Unassembled mask (1=good, 0=bad)
        self.assembled_mask - Assembled mask (1-good, 0=bad)
    """
    def __init__(self, fname=None, **kwargs):
        self.det = <c_det.detector*> calloc(1, sizeof(c_det.detector))
        if fname is not None:
            self.parse(fname, **kwargs)

    def parse(self, fname, **kwargs):
        """ Parse Dragonfly detector from file

        File can either be in the HDF5 or ASCII format
        """
        self.det.fname = <char*> malloc(len(fname)+1)
        strcpy(self.det.fname, bytes(fname, 'utf-8'))
        if h5py.is_hdf5(fname):
            self._parse_h5det(**kwargs)
        elif os.path.splitext(fname)[1] == '.h5':
            fheader = np.fromfile(fname, '=c', count=8)
            if fheader == chr(137)+'HDF\r\n'+chr(26)+'\n':
                self._parse_h5det(**kwargs)
            else:
                self._parse_asciidet(**kwargs)
        else:
            self._parse_asciidet(**kwargs)

    def free(self):
        if self.det == NULL:
            return
        if self.det.qvals != NULL:
            free(self.det.qvals)
            self.det.qvals = NULL
        if self.det.corr != NULL:
            free(self.det.corr)
            self.det.corr = NULL
        if self.det.raw_mask != NULL:
            free(self.det.raw_mask)
            self.det.raw_mask = NULL
        if self.det.fname != NULL:
            free(self.det.fname)
            self.det.fname = NULL
        free(self.det)
        self.det = NULL

    def qmax(self):
        cdef int t, d
        cdef double qsq, qmax = 0
        for t in range(self.det.num_pix):
            qsq = 0
            for d in range(3):
                qsq += self.det.qvals[t*3 + d]**2

            if qsq > qmax:
                qmax = qsq
        return sqrt(qmax)

    def _check_header(self):
        with open(self.fname, 'r') as fptr:
            line = fptr.readline().rstrip().split()
        if len(line) != 1:
            self.det.detd = float(line[1])
            self.det.ewald_rad = float(line[2])
        else:
            raise ValueError('Need 3 values on header line: num_pix, detd_pix, ewald_rad_vox')

    def _parse_asciidet(self, norm=True, rtype='3d'):
        self._check_header()
        dframe = pandas.read_csv(
            self.fname,
            delim_whitespace=True, skiprows=1, engine='c', header=None,
            names=['qx', 'qy', 'qz', 'corr', 'mask'],
            dtype={'qx':'f8', 'qy':'f8', 'qz':'f8', 'corr':'f8', 'mask':'u1'})
        qx, qy, qz, np_corr = tuple([np.array(dframe[key]) # pylint: disable=C0103
                                       for key in ['qx', 'qy', 'qz', 'corr']])
        self.det.num_pix = qx.shape[0]
        if norm:
            np_corr /= np_corr.mean()
        if rtype == '2d':
            self._conv_2d(qx, qy, qz)

        cdef int t, d
        cdef double[:,:] qvals = np.copy(np.array([qx, qy, qz]).T)
        cdef double[:] corr = np_corr
        cdef uint8_t[:] raw_mask = np.array(dframe['mask']).astype('u1')
        self.det.qvals = <double*> malloc(self.num_pix * 3 * sizeof(double))
        self.det.corr = <double*> malloc(self.num_pix * sizeof(double))
        self.det.raw_mask = <uint8_t*> malloc(self.num_pix * sizeof(uint8_t))

        for t in range(self.num_pix):
            self.det.corr[t] = corr[t]
            self.det.raw_mask[t] = raw_mask[t]
            for d in range(3):
                self.det.qvals[t*3 + d] = qvals[t, d]

    def _parse_h5det(self, norm=True, rtype='3d'):
        fptr = h5py.File(self.fname, 'r')
        qx, qy, qz = fptr['qx'][:], fptr['qy'][:], fptr['qz'][:]
        cdef double[:] corr = fptr['corr'][:].ravel()
        cdef uint8_t[:] raw_mask = fptr['mask'][:].astype('u1')
        self.det.detd = fptr['detd'][()]
        self.det.ewald_rad = fptr['ewald_rad'][()]
        if 'background' in fptr:
            bg_present = True
            background = fptr['background'][:].ravel()
        else:
            bg_present = False
        fptr.close()

        if norm:
            np_corr = np.asarray(corr)
            np_corr /= np_corr.mean()
        if rtype == '2d':
            self._conv_2d(qx, qy, qz)

        self.det.num_pix = qx.shape[0]
        cdef double[:,:] qvals = np.ascontiguousarray(np.array([qx, qy, qz]).T).reshape(-1,3)

        self.det.qvals = <double*> malloc(self.num_pix * 3 * sizeof(double))
        self.det.corr = <double*> malloc(self.num_pix * sizeof(double))
        self.det.raw_mask = <uint8_t*> malloc(self.num_pix * sizeof(uint8_t))
        if bg_present:
            self.det.background = <double*> malloc(self.num_pix * sizeof(double))

        cdef int t, d
        for t in range(self.num_pix):
            self.det.corr[t] = corr[t]
            self.det.raw_mask[t] = raw_mask[t]
            if bg_present:
                self.det.background[t] = background[t]

            for d in range(3):
                self.det.qvals[t*3 + d] = qvals[t, d]

    def _conv_2d(self, qx, qy, qz):
        if qz.mean() > 0:
            qx *= self.detd / (self.ewald_rad - qz) # pylint: disable=C0103
            qy *= self.detd / (self.ewald_rad - qz) # pylint: disable=C0103
        else:
            qx *= self.detd / (self.ewald_rad + qz) # pylint: disable=C0103
            qy *= self.detd / (self.ewald_rad + qz) # pylint: disable=C0103
        qz[:] = 0

    @property
    def fname(self): return (<bytes> self.det.fname).decode()
    @property
    def num_pix(self): return self.det.num_pix
    @property
    def detd(self): return self.det.detd
    @detd.setter
    def detd(self, value): self.det.detd = float(value)
    @property
    def ewald_rad(self): return self.det.ewald_rad
    @ewald_rad.setter
    def ewald_rad(self, value): self.det.ewald_rad = float(value)
    @property
    def corr(self): return np.asarray(<double[:self.num_pix]>self.det.corr) if self.det.corr != NULL else None
    @corr.setter
    def corr(self, arr):
        if len(arr.shape) != 1 or arr.dtype != 'f8':
            raise ValueError('corr must be  1D array of float64 dtype')
        if self.det.num_pix > 0 and (arr.shape[0] != self.det.num_pix):
            raise ValueError('num_pix mismatch with other data (%d vs %d)'%(arr.shape[0], self.det.num_pix))

        self.det.corr = <double*> malloc(arr.size * sizeof(double))
        for i in range(arr.size):
            self.det.corr[i] = arr[i]
        if self.det.num_pix == 0:
            self.det.num_pix = arr.shape[0]
    @property
    def raw_mask(self): return np.asarray(<uint8_t[:self.num_pix]>self.det.raw_mask) if self.det.raw_mask != NULL else None
    @raw_mask.setter
    def raw_mask(self, arr):
        if len(arr.shape) != 1 or arr.dtype != 'u1':
            raise ValueError('raw_mask must be  1D array of uint8 dtype')
        if self.det.num_pix > 0 and (arr.shape[0] != self.det.num_pix):
            raise ValueError('num_pix mismatch with other data (%d vs %d)'%(arr.shape[0], self.det.num_pix))

        self.det.raw_mask = <uint8_t*> malloc(arr.size * sizeof(uint8_t))
        for i in range(arr.size):
            self.det.raw_mask[i] = arr[i]
        if self.det.num_pix == 0:
            self.det.num_pix = arr.shape[0]
    @property
    def qvals(self): return np.asarray(<double[:3*self.num_pix]>self.det.qvals).reshape(-1, 3) if self.det.qvals != NULL else None
    @qvals.setter
    def qvals(self, arr):
        if len(arr.shape) != 2 or arr.shape[1] != 3:
            raise ValueError('qvals must be  2D array of shape (N, 3)')
        if self.det.num_pix > 0 and (arr.shape[0] != self.det.num_pix):
            raise ValueError('num_pix mismatch with other data (%d vs %d)'%(arr.shape[0], self.det.num_pix))
        if arr.dtype != 'f8':
            raise TypeError('qvals must be double precision floats (float64)')

        self.det.qvals = <double*> malloc(arr.size * sizeof(double))
        for i in range(arr.size):
            self.det.qvals[i] = arr.ravel()[i]
        if self.det.num_pix == 0:
            self.det.num_pix = arr.shape[0]
    @property
    def background(self): return np.asarray(<double[:self.num_pix]>self.det.background) if self.det.background != NULL else None
    @background.setter
    def background(self, arr):
        if len(arr.shape) != 1 or arr.dtype != 'f8':
            raise ValueError('corr must be  1D array of float64 dtype')
        if self.det.num_pix > 0 and (arr.shape[0] != self.det.num_pix):
            raise ValueError('num_pix mismatch with other data (%d vs %d)'%(arr.shape[0], self.det.num_pix))

        self.det.background = <double*> malloc(arr.size * sizeof(double))
        for i in range(arr.size):
            self.det.background[i] = arr[i]
        if self.det.num_pix == 0:
            self.det.num_pix = arr.shape[0]

class Detector(CDetector):
    def __init__(self, fname=None, **kwargs):
        '''
        Additional methods of python Detector class:
            assemble_frame(data, zoomed=False, sym=False)
            calc_from_coords()
            remask(qradius)
            write(fname)
        '''
        super(Detector, self).__init__(fname, **kwargs)
        self._sym_shape = None

    def parse(self, fname, mask_flag=True, keep_mask_1=True):
        super(Detector, self).parse(fname)
        self._process_det(mask_flag=mask_flag, keep_mask_1=keep_mask_1)

    def write(self, fname):
        """ Write Dragonfly detector to file

        If h5py is available and the file name as a '.h5' extension,
        an HDF5 detector will be written, otherwise an ASCII file will be generated.

        Note that the background array can only be stored in an HDF5 detector
        """
        if not (hasattr(self, "qvals") and
                hasattr(self, "corr") and
                hasattr(self, "raw_mask") and
                hasattr(self, "detd") and
                hasattr(self, "ewald_rad")):
            raise AttributeError('Detector attributes not populated. Cannot write to file')

        if os.path.splitext(fname)[1] == '.h5':
            self._write_h5det(fname)
        else:
            self._write_asciidet(fname)

    def assemble_frame(self, data, zoomed=False, sym=False, avg=False):
        ''' Assemble given raw image

        Arguments:
            data - array of num_pix values
            zoomed (bool) - Restrict assembled image to non-masked pixels
            sym (bool) - Centro-symmetrize image
            avg (bool) - Average assembled image

        Returns:
            Numpy masked array representing assembled image
        '''
        if sym:
            self._init_sym()
            img = ma.masked_array(np.zeros(self._sym_shape, dtype='f8'), mask=1-self._sym_mask)
            np.add.at(img, (self._sym_x, self._sym_y), data*self.mask)
            np.add.at(img, (self._sym_fx, self._sym_fy), data*self.mask)

            if avg:
                countimg = np.zeros(self._sym_shape, dtype='f8')
                np.add.at(countimg, (self._sym_x, self._sym_y), self.mask)
                np.add.at(countimg, (self._sym_fx, self._sym_fy), self.mask)
                img.data[countimg>0] /= countimg[countimg>0]
            else:
                img.data[self._sym_bothgood] /= 2.

            if zoomed:
                b = self._sym_zoom_bounds
                return img[b[0]:b[1], b[2]:b[3]]
        else:
            img = ma.masked_array(np.zeros(self.frame_shape, dtype='f8'), mask=1-self.assembled_mask)
            np.add.at(img, (self.x, self.y), data*self.mask)
            if avg:
                countimg = np.zeros(self.frame_shape, dtype='f8')
                np.add.at(countimg, (self.x, self.y), self.mask)
                img.data[countimg>0] /= countimg[countimg>0]
            if zoomed:
                b = self.zoom_bounds
                return img[b[0]:b[1], b[2]:b[3]]
        return img

    def calc_from_coords(self, pol='x'):
        ''' Calculate essential detector attributes from pixel coordinates

        Needs:
            cx, cy, detd, ewald_rad
        Calculates:
            qvals and corr
        '''
        try:
            val = self.cx + self.cy
            if self.detd == 0. or self.ewald_rad == 0.:
                raise AttributeError
        except AttributeError:
            print('Need cx, cy, detd and ewald_rad to be defined')
            print('detd must have same units as cx and cy')
            print('ewald_rad should be in voxel units')
            return

        self.cx = self.cx.ravel()
        self.cy = self.cy.ravel()

        fac = np.sqrt(self.cx**2 + self.cy**2 + self.detd**2)
        qvals = np.empty(self.cx.shape + (3,))
        qvals[:,0] = self.cx * self.ewald_rad / fac
        qvals[:,1] = self.cy * self.ewald_rad / fac
        qvals[:,2] = self.ewald_rad * (self.detd/fac - 1.)
        if pol.lower() == 'x':
            corr = self.detd / fac**3 * (1. - self.cx**2 / fac**2)
        elif pol.lower() == 'y':
            corr = self.detd / fac**3 * (1. - self.cy**2 / fac**2)
        else:
            corr = self.detd / fac**3

        # This assignment forces a copy
        self.qvals = qvals
        self.corr = corr

    def remask(self, qradius):
        ''' Remask detector with given q-radius

        Sets mask value of all good pixels (mask==0) with q-radius greater than specified
        to be irrelevant (mask==1).
        This is useful when doing coarse orientational alignment
        '''
        if self._qrad is None:
            self._qrad = np.linalg.norm(self.qvals, axis=1)
        self.raw_mask[(self.raw_mask == 0) & (self._qrad > qradius)] = 1

    def parse_background(self, fname):
        if h5py.ishdf5(fname):
            with h5py.File(fname, 'r') as fptr:
                self.background = fptr['background'][:].ravel()
        else:
            self.background = np.fromfile(fname)

    def _process_det(self, mask_flag, keep_mask_1):
        self.shape = self.corr.shape
        if mask_flag:
            mask = np.copy(self.raw_mask)
            if keep_mask_1:
                mask[mask == 1] = 0 # To keep both 0 and 1
                mask = mask // 2 # To keep both 0 and 1
            else:
                mask[mask == 2] = 1 # To keep only mask==0
            mask = 1 - mask
        else:
            self.raw_mask = np.zeros(self.shape, dtype='u1')
            mask = np.ones(self.shape, dtype='u1')

        if self.qvals[:,2].mean() > 0:
            self.cx = self.qvals[:,0] * self.detd / (self.ewald_rad - self.qvals[:,2]) # pylint: disable=C0103
            self.cy = self.qvals[:,1] * self.detd / (self.ewald_rad - self.qvals[:,2]) # pylint: disable=C0103
        else:
            self.cx = self.qvals[:,0] * self.detd / (self.ewald_rad + self.qvals[:,2]) # pylint: disable=C0103
            self.cy = self.qvals[:,1] * self.detd / (self.ewald_rad + self.qvals[:,2]) # pylint: disable=C0103
        self.x = np.round(self.cx - self.cx.min()).astype('i4')
        self.y = np.round(self.cy - self.cy.min()).astype('i4')
        self.mask = mask.ravel().astype('bool')
        self._init_assem()

    def _init_assem(self):
        '''Calculate attributes given self.x and self.y'''
        self.frame_shape = (self.x.max()+1, self.y.max()+1)

        self.assembled_mask = np.zeros(self.frame_shape, dtype='u1')
        self.assembled_mask[self.x, self.y] = self.mask
        self.assembled_mask = np.sign(self.assembled_mask)

        xsel = self.x[self.mask]
        ysel = self.y[self.mask]
        self.zoom_bounds = (xsel.min(), xsel.max()+1, ysel.min(), ysel.max()+1)

    def _write_asciidet(self, fname):
        print('Writing ASCII detector file')
        if self.background is not None:
            print('WARNING! ASCII detector files cannot save background')
        qx = self.qvals[:,0].ravel()
        qy = self.qvals[:,1].ravel()
        qz = self.qvals[:,2].ravel()
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
            fptr['qx'] = self.qvals[:,0].ravel().astype('f8')
            fptr['qy'] = self.qvals[:,1].ravel().astype('f8')
            fptr['qz'] = self.qvals[:,2].ravel().astype('f8')
            fptr['corr'] = self.corr.ravel().astype('f8')
            fptr['mask'] = self.raw_mask.ravel().astype('u1')
            fptr['detd'] = float(self.detd)
            fptr['ewald_rad'] = float(self.ewald_rad)
            if self.background is not None:
                fptr['background'] = self.background.ravel().astype('f8')

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
        np.add.at(self._sym_mask, (self._sym_x, self._sym_y), self.mask)
        np.add.at(self._sym_mask, (self._sym_fx, self._sym_fy), self.mask)
        self._sym_bothgood = (self._sym_mask == 2)
        self._sym_mask = np.sign(self._sym_mask)

        mask = self.mask
        xsel = np.concatenate((self._sym_x[mask.astype('bool')], self._sym_fx[mask.astype('bool')]))
        ysel = np.concatenate((self._sym_y[mask.astype('bool')], self._sym_fy[mask.astype('bool')]))
        self._sym_zoom_bounds = (xsel.min(), xsel.max()+1, ysel.min(), ysel.max()+1)

    @property
    def coords_xy(self):
        '''Return 2D pixel coordinates'''
        return self.cx, self.cy

    @property
    def indices_xy(self):
        '''Return 2D integer coordinates (for assembly)
        Corner of the detector at (0,0)'''
        return self.x, self.y

