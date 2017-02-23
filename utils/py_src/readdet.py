import sys
import os
import numpy as np

class Det_reader():
    def __init__(self, det_fname, detd, ewald_rad, mask_flag=False):
        self.det_fname = det_fname
        self.detd = detd
        self.ewald_rad = ewald_rad
        self.init_geom(mask_flag)

    def init_geom(self, mask_flag):
        sys.stderr.write('Reading detector file...')
        if mask_flag:
            sys.stderr.write('with mask...')
            self.qx, self.qy, self.qz, mask = np.loadtxt(self.det_fname, usecols=(0,1,2,4), skiprows=1, unpack=True)
            #mask[mask==2] = 1 # To keep only mask==0
            mask[mask==1] = 0 # To keep both 0 and 1
            mask = mask / 2 # To keep both 0 and 1
            mask = 1 - mask
        else:
            self.qx, self.qy, self.qz = np.loadtxt(self.det_fname, usecols=(0,1,2), skiprows=1, unpack=True)
            mask = np.ones(self.qx.shape)
        sys.stderr.write('done\n')
        
        self.cx = self.qx*self.detd/(self.qz+self.ewald_rad)
        self.cy = self.qy*self.detd/(self.qz+self.ewald_rad)
        self.x = np.round(self.cx - self.cx.min()).astype('i4')
        self.y = np.round(self.cy - self.cy.min()).astype('i4')
        
        self.frame_shape = (self.x.max()+1, self.y.max()+1)
        self.mask = np.ones(self.frame_shape)
        self.mask[self.x, self.y] = mask.flatten()
        self.raw_mask = mask

