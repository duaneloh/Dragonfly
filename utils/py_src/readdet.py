import sys
import os
import numpy as np

class Det_reader():
    """Dragonfly detector file reader
    The detector file format is specified in github.com/duaneloh/Dragonfly/wiki
    This class reads the file and provides numpy arrays which can be used for
    further processing.
    
    __init__ arguments:
        det_fname (string) - Path to detector file
        detd_pix (float, optional) - Detector distance in pixels (detd/pixsize)
        ewald_rad (float, optional) - Ewald sphere radius in voxels. If in doubt, = detd_pix
        mask_flag (bool, optional) - Whether to read the mask column for each pixel
    For the new detector file, detd_pix and ewald_rad numbers are read from the file
    but for the old file, they are required.

    On initialization, it produces the following numpy arrays (each of length num_pix)
        self.qx, self.qy, self.qz - Voxel space coordinates (origin at (0,0,0))
        self.cx, self.cy - Floating point 2D coordinates (origin at (0,0))
        self.x, self.y - Integer and shifted 2D coordinates (corner at (0,0))
        self.mask - Assembled mask
        self.raw_mask - Unassembled mask as stored in detector file
        self.unassembled_mask - Unassembled mask (1=good, 0=bad)
    """
    def __init__(self, det_fname, detd_pix=None, ewald_rad=None, mask_flag=False, keep_mask_1=True):
        self.det_fname = det_fname
        self.detd = detd_pix
        self.ewald_rad = ewald_rad
        self._check_header()
        self._init_geom(mask_flag, keep_mask_1)

    def _check_header(self):
        with open(self.det_fname, 'r') as f:
            line = f.readline().rstrip().split()
        if len(line) > 1:
            self.detd = float(line[1])
            self.ewald_rad = float(line[2])
        else:
            if self.detd is None:
                raise TypeError('Old type detector file. Need detd_pix')
            if self.ewald_rad is None:
                raise TypeError('Old type detector file. Need ewald_rad')

    def _init_geom(self, mask_flag, keep_mask_1):
        """ (Internal) Detector file parser
        Arguments:
            mask_flag (bool, optional) - Whether to read the mask column
        """
        sys.stderr.write('Reading %s...'%self.det_fname)
        if mask_flag:
            sys.stderr.write('with mask...')
            self.qx, self.qy, self.qz, self.corr, raw_mask = np.loadtxt(self.det_fname, skiprows=1, unpack=True)
            mask = np.copy(raw_mask).astype('u1')
            if keep_mask_1:
                mask[mask==1] = 0 # To keep both 0 and 1
                mask = mask / 2 # To keep both 0 and 1
            else:
                mask[mask==2] = 1 # To keep only mask==0
            mask = 1 - mask
        else:
            self.qx, self.qy, self.qz, self.corr = np.loadtxt(self.det_fname, usecols=(0,1,2,3), skiprows=1, unpack=True)
            raw_mask = np.zeros(self.qx.shape)
            mask = np.ones(self.qx.shape, dtype='u1')
        sys.stderr.write('done\n')
        
        self.cx = self.qx*self.detd/(self.qz+self.ewald_rad)
        self.cy = self.qy*self.detd/(self.qz+self.ewald_rad)
        self.x = np.round(self.cx - self.cx.min()).astype('i4')
        self.y = np.round(self.cy - self.cy.min()).astype('i4')
        
        self.frame_shape = (self.x.max()+1, self.y.max()+1)
        self.mask = np.ones(self.frame_shape, dtype='u1')
        self.mask[self.x, self.y] = mask.flatten()
        self.unassembled_mask = mask
        self.raw_mask = raw_mask.astype('u1')

