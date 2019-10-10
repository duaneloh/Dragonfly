import sys
import os
import numpy as np
import pandas
import h5py

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
cimport numpy as np
from . cimport iterate as c_iterate
from . cimport params as c_params
from .iterate cimport Iterate
from .detector cimport CDetector, detector
from .model cimport Model
from .emcfile cimport CDataset, dataset
from .quaternion cimport Quaternion
from .params cimport EMCParams

cdef class Iterate:
    def __init__(self):
        self.iter = <c_iterate.iterate*> calloc(1, sizeof(c_iterate.iterate))
        self.iter.par = <c_params.params*> calloc(1, sizeof(c_params.params))

    def set_model(self, Model model):
        self.iter.mod = model.mod

    def set_data(self, CDataset in_dset):
        cdef int total = 0
        cdef dataset *curr = in_dset.dset

        # Calculate total number of frames
        while curr != NULL:
            curr.num_offset = total
            total += curr.num_data
            curr = curr.next

        # Check whether it matches any existing value
        if self.iter.tot_num_data == 0:
            self.iter.tot_num_data = total
        elif self.iter.tot_num_data != total:
            raise ValueError('Mismatched total number of frames')

        self.iter.dset = in_dset.dset

        # Generate list of unique detectors
        self._gen_detlist()
        
        print('Calculating frame_counts')
        c_iterate.calc_frame_counts(self.iter)

    def set_quat(self, Quaternion quaternion):
        self.iter.quat = quaternion.quat

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
        cdef double mean_scale

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
                self.iter.scale[d] /= mean_scale
        else:
            for d in range(self.iter.tot_num_data):
                if self.iter.blacklist[d] == 0:
                    self.iter.scale[d] /= mean_scale

        self.iter.rms_change *= mean_scale

    def calc_beta(self, setval=None):
        if self.iter.dset == NULL:
            raise AttributeError('Add data first before calculating beta for each frame')

        cdef double start

        if setval is not None:
            start = setval
        else:
            start = -1.

        c_iterate.calc_beta(start, self.iter)

    def _gen_detlist(self):
        cdef int i, d, ind
        cdef dataset *curr = self.iter.dset

        fnames = []
        while curr != NULL:
            fnames.append(curr.det.fname)
            curr = curr.next
        _, indices, mapping = np.unique(fnames, return_index=True, return_inverse=True)
        self.iter.num_det = len(indices)

        self.iter.det = <detector*> calloc(len(indices), sizeof(detector))
        for i, ind in enumerate(indices):
            curr = self.iter.dset
            for d in range(ind):
                curr = curr.next
            memcpy(&self.iter.det[i], curr.det, sizeof(detector))

        self.iter.det_mapping = <int*> malloc(len(mapping) * sizeof(int))
        for i, d in enumerate(mapping):
            self.iter.det_mapping[i] = d

    @staticmethod
    def calculate_size(qmax):
        return int(2 * np.ceil(qmax) + 3)

    @property
    def tot_num_data(self): return self.iter.tot_num_data
    @tot_num_data.setter
    def tot_num_data(self, val): self.iter.tot_num_data = val
    @property
    def fcounts(self): return np.asarray(<int[:self.tot_num_data]>self.iter.fcounts) if self.iter.fcounts != NULL else None
    @property
    def scale(self): return np.asarray(<double[:self.tot_num_data]>self.iter.scale) if self.iter.scale != NULL else None
    @property
    def bgscale(self): return np.asarray(<double[:self.tot_num_data]>self.iter.bgscale) if self.iter.bgscale != NULL else None
    @property
    def beta(self): return np.asarray(<double[:self.tot_num_data]>self.iter.beta) if self.iter.beta != NULL else None
    @property
    def beta_start(self): return np.asarray(<double[:self.tot_num_data]>self.iter.beta_start) if self.iter.beta_start != NULL else None
    @property
    def blacklist(self): return np.asarray(<uint8_t[:self.tot_num_data]>self.iter.blacklist) if self.iter.blacklist != NULL else None

    @property
    def num_det(self): return self.iter.num_det
    @property
    def dets(self):
        retval = [None for d in range(self.iter.num_det)]
        for d in range(self.iter.num_det):
            curr = CDetector()
            curr.free()
            curr.det = &self.iter.det[d]
            retval[d] = curr
        return retval
    @property
    def model(self):
        retval = Model()
        retval.free()
        retval.mod = self.iter.mod
        return retval
    @property
    def data(self):
        retval = CDataset()
        retval.free()
        retval.dset = self.iter.dset
        return retval
    @property
    def params(self):
        retval = EMCParams()
        retval.free()
        retval.par = self.iter.par
        return retval
