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
        detd_pix (float) - Detector distance in pixels (detd/pixsize)
        ewald_rad (float) - Ewald sphere radius in voxels. If in doubt, = detd_pix
        mask_flag (bool, optional) - Whether to read the mask column for each pixel

    On initialization, it produces the following numpy arrays (each of length num_pix)
        self.qx, self.qy, self.qz - Voxel space coordinates (origin at (0,0,0))
        self.cx, self.cy - Floating point 2D coordinates (origin at (0,0))
        self.x, self.y - Integer and shifted 2D coordinates (corner at (0,0))
        self.mask - Assembled mask
        self.raw_mask - Unassembled mask as stored in detector file
        self.unassembled_mask - Unassembled mask (1=good, 0=bad)
    """
    def __init__(self, det_fname, detd_pix, ewald_rad, mask_flag=False):
        self.det_fname = det_fname
        self.detd = detd_pix
        self.ewald_rad = ewald_rad
        self._init_geom(mask_flag)

    def _init_geom(self, mask_flag):
        """ (Internal) Detector file parser
        Arguments:
            mask_flag (bool, optional) - Whether to read the mask column
        """
        sys.stderr.write('Reading detector file...')
        if mask_flag:
            sys.stderr.write('with mask...')
            self.qx, self.qy, self.qz, self.corr, raw_mask = np.loadtxt(self.det_fname, skiprows=1, unpack=True)
            #mask[mask==2] = 1 # To keep only mask==0
            mask = np.copy(raw_mask).astype('u1')
            mask[mask==1] = 0 # To keep both 0 and 1
            mask = mask / 2 # To keep both 0 and 1
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

