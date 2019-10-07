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
from .detector cimport CDetector, detector
from .model cimport Model
from .emcfile cimport CDataset
from .quaternion cimport Quaternion

cdef class Iterate:
    def __init__(self):
        self.iter = <c_iterate.iterate*> calloc(1, sizeof(c_iterate.iterate))

    def set_detectors(self, det_list):
        cdef int d
        cdef CDetector in_det
        if self.iter.det != NULL:
            free(self.iter.det)

        if isinstance(det_list, CDetector):
            in_det = det_list
            print('Setting single detector')
            self.iter.det = in_det.det
            self.iter.num_det = 1
        else:
            print('%d detectors in iterate' % len(det_list))
            self.iter.num_det = len(det_list)
            self.iter.det = <detector*> malloc(self.iter.num_det * sizeof(detector))
            for d in range(self.iter.num_det):
                in_det = det_list[d]
                memcpy(&self.iter.det[d], in_det.det, sizeof(detector))

    def set_model(self, Model model):
        self.iter.mod = model.mod

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

    def parse_blacklist(self, fname, sel_string=None, refresh=False):
        '''Generate blacklist from file and selection string

        Blacklist file contains one number (0 or 1) per line for each frame indicating whether
        the frame is blacklisted (1) or considered good (0).

        On top of that for dataset splitting, one can provide a selection string, either
        'odd_only' or 'even_only' to take only half of the good frames.
        '''
        if self.iter.tot_num_data == 0:
            raise AttributeError('Need to define tot_num_data before generating blacklist')

        cdef int d
        cdef uint8_t curr
        cdef uint8_t[:] arr
        if self.iter.blacklist == NULL:
            self.iter.blacklist = <uint8_t*> calloc(self.iter.tot_num_data, sizeof(uint8_t))
        elif refresh:
            free(self.iter.blacklist)
            self.iter.blacklist = <uint8_t*> calloc(self.iter.tot_num_data, sizeof(uint8_t))
        else:
            print('Applying changes to current blacklist')

        if os.path.isfile(fname):
            arr = pandas.read_csv(fname, header=None, squeeze=True, dtype='u1').to_numpy()
            if arr.shape[0] != self.iter.tot_num_data:
                raise ValueError('Mismatched number of frames in blacklist file')
            memcpy(self.iter.blacklist, &arr[0], self.iter.tot_num_data*sizeof(uint8_t))

        if sel_string is None:
            return

        if sel_string == 'odd_only':
            curr = 0
        elif sel_string == 'even_only':
            curr = 1
        else:
            raise ValueError('Unrecognized sel_string, %s' % sel_string)

        for d in range(self.iter.tot_num_data):
            if self.iter.blacklist[d] == 0:
                self.iter.blacklist[d] = curr
                curr = 1 - curr

    def normalize_scale(self):
        cdef long x, d

        blist = self.blacklist
        if blist is None:
            mean_scale = self.scale.mean()
        else:
            mean_scale = self.scale[blist==0].mean()

        if self.iter.mod != NULL and self.iter.mod.model1 != NULL:
            for x in range(self.iter.mod.vol):
                self.iter.mod.model1[x] *= mean_scale

        if blist is None:
            for x in range(self.iter.tot_num_data):
                self.scale[d] /= mean_scale
        else:
            for d in range(self.iter.tot_num_data):
                if blist[d] == 0:
                    self.scale[d] /= mean_scale

        self.iter.rms_change *= mean_scale

    @staticmethod
    def calculate_size(qmax):
        return int(2 * np.ceil(qmax) + 3)

    @property
    def tot_num_data(self): return self.iter.tot_num_data
    @tot_num_data.setter
    def tot_num_data(self, val): self.iter.tot_num_data = val
    @property
    def scale(self): return np.asarray(<double[:self.tot_num_data]>self.iter.scale) if self.iter.scale != NULL else None
    @property
    def blacklist(self): return np.asarray(<uint8_t[:self.tot_num_data]>self.iter.blacklist) if self.iter.blacklist != NULL else None

    @property
    def num_det(self): return self.iter.num_det
    @property
    def dets(self):
        retval = [None for d in range(self.iter.num_det)]
        for d in range(self.iter.num_det):
            curr = CDetector()
            curr.det = &self.iter.det[d]
            retval[d] = curr
        return retval
    @property
    def model(self):
        retval = Model()
        retval.mod = self.iter.mod
        return retval
