import sys
import os
import numpy as np
import pandas
import h5py

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
cimport numpy as np
from . cimport iterate as c_iterate
from .iterate cimport Iterate
from .detector cimport CDetector
from .model cimport Model
from .emcfile cimport CDataset
from .quaternion cimport Quaternion

cdef class Iterate:
    def __init__(self):
        self.iter = <c_iterate.iterate*> calloc(1, sizeof(c_iterate.iterate))

    def parse_scale(self, fname, bg=False):
        if h5py.is_hdf5(fname):
            with h5py.File(fname, 'r') as f:
                scale = f['scale'][:]
        else:
            scale = pandas.read_csv(fname, header=None).array.ravel()
        
        cdef double[:] scale_view = scale

        if self.iter.tot_num_data == 0:
            self.iter.tot_num_data = scale_view.shape[0]
        elif self.iter.tot_num_data < scale_view.shape[0]:
            print('More scale factors than required. Taking only first', self.iter.tot_num_data)
        elif self.iter.tot_num_data > scale_view.shape[0]:
            print('Insufficient scale factors in file. Setting rest to unity')
            scale = np.append(scale, np.ones(self.iter.tot_num_data - scale_view.shape[0]))
            scale_view = scale

        if bg:
            self.iter.bgscale = <double*> malloc(self.iter.tot_num_data * sizeof(double))
            memcpy(self.iter.bgscale, &scale_view[0], self.iter.tot_num_data * sizeof(double))
        else:
            self.iter.scale = <double*> malloc(self.iter.tot_num_data * sizeof(double))
            memcpy(self.iter.scale, &scale_view[0], self.iter.tot_num_data * sizeof(double))

    def parse_blacklist(self, fname, sel_string=None):
        '''Generate blacklist from file and selection string
        
        Blacklist file contains one number (0 or 1) per line for each frame indicating whether
        the frame is blacklisted (1) or considered good (0).
        
        On top of that for dataset splitting, one can provide a selection string, either
        'odd_only' or 'even_only' to take only half of the good frames.
        '''
        cdef uint8_t[:] arr
        if os.path.isfile(fname):
            arr = pandas.read_csv(fname, header=None, squeeze=True, dtype='u1').array
            self.iter.blacklist = <uint8_t*> malloc(arr.shape[0] * sizeof(uint8_t))
            memcpy(&self.iter.blacklist, &arr[0], arr.shape[0])

        if sel_string is 'odd_only':
            self.blacklist[self.blacklist==0][0::2] = 1
        elif sel_string is 'even_only':
            self.blacklist[self.blacklist==0][1::2] = 1

    @staticmethod
    def calculate_size(qmax):
        return int(2 * np.ceil(qmax) + 3)


    @property
    def tot_num_data(self): return self.iter.tot_num_data
    @property
    def scale(self): return np.asarray(<double[:self.tot_num_data]>self.iter.scale) if self.iter.scale != NULL else None
'''
    def normalize_scale(self, frames):
        blist = frames.blacklist
        mean_scale = self.scale[blist==0].mean()
        self.iter.mod.model1 *= mean_scale
        self.scale[blist==0] /= mean_scale
        self.rms_change *= mean_scale
'''
