'''Module containing EMCReader class to parse 2D stack files'''

from __future__ import print_function
import sys
import numpy as np
try:
    import h5py
    HDF5_MODE = True
except ImportError:
    HSD5_MODE = False

class VolReader(object):
    def __init__(self, stack_fname, size):
        self.fname = stack_fname
        self.size = size
        if HDF5_MODE and h5py.is_hdf5(stack_fname):
            with h5py.File(stack_fname, 'r') as f:
                self.stack = f['intens'][:]
        else:
            self.stack = np.fromfile(stack_fname).reshape(-1, size, size)
        
        self.flist = [{'fname': stack_fname,
            'num_data': self.stack.shape[0],
            'num_pix': size*size}]
        self.num_frames = self.stack.shape[0]

    def get_frame(self, num, raw=True, **kwargs):
        return self.stack[num]

    def get_powder(self, raw=True, **kwargs):
        return self.stack.mean(0)
