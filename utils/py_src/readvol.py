'''Module containing EMCReader class to parse 2D stack .bin files'''

from __future__ import print_function
import sys
import numpy as np

class VolReader(object):
    def __init__(self, stack_fname, size):
        self.fname = stack_fname
        self.size = size
        self.stack = np.fromfile(stack_fname).reshape(-1, size, size)
        
        self.flist = [{'fname': stack_fname,
            'num_data': self.stack.shape[0],
            'num_pix': size*size}]
        self.num_frames = self.stack.shape[0]

    def get_frame(self, num, raw=True):
        return self.stack[num]

    def get_powder(self, raw=True):
        return self.stack.mean(0)
