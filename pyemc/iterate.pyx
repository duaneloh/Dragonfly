import numpy as np
cimport numpy as np
from scipy import sparse

from libc.stdlib cimport malloc, free

cimport decl
from dataset cimport dataset
from detector cimport detector
from iterate cimport iterate
from params cimport params

cdef class iterate:
    def __init__(self, allocate=True):
        if allocate:
            self._alloc()
        else:
            self.iterate = NULL

    def _alloc(self):
        self.iterate = <decl.iterate*> malloc(sizeof(decl.iterate))
        self.iterate.size = -1
        self.iterate.model1 = NULL
        self.iterate.model2 = NULL
        self.iterate.inter_weight = NULL
        self.iterate.scale = NULL
        self.iterate.rescale = NULL
        self.iterate.rel_quat = NULL

    def generate_iterate(self, config_fname, double qmax, params param, detector det, dataset dset, continue_flag=False, config_section=b'emc'):
        cdef char* c_config_fname = config_fname
        cdef char* c_config_section = config_section
        ret = decl.generate_iterate(c_config_fname, c_config_section, int(continue_flag), qmax, param.param, det.det, dset.dset, self.iterate)
        assert ret == 0

    def calculate_size(self, double qmax):
        decl.calculate_size(qmax, self.iterate)
        return self.size

    def parse_scale(self, fname):
        cdef char* c_fname = fname
        return decl.parse_scale(c_fname, self.iterate)

    def calc_scale(self, dataset dset, detector det):
        decl.calc_scale(dset.dset, det.det, self.iterate)

    def normalize_scale(self, dataset dset):
        decl.normalize_scale(dset.dset, self.iterate)

    def parse_input(self, fname, double mean, int rank=0):
        cdef char* c_fname = fname
        decl.parse_input(c_fname, mean, rank, 42, self.iterate)

    def parse_rel_quat(self, fname, int num_rot_coarse, parse_prob=False):
        cdef char *c_fname = fname
        cdef int c_parse_prob = int(parse_prob)
        return decl.parse_rel_quat(c_fname, num_rot_coarse, c_parse_prob, self.iterate)

    def free_iterate(self):
        decl.free_iterate(self.iterate)
        self.iterate = NULL

    @property
    def size(self): return self.iterate.size if self.iterate != NULL else None
    @property
    def center(self): return self.iterate.center if self.iterate != NULL else None
    @property
    def vol(self): return self.iterate.vol if self.iterate != NULL else None
    @property
    def tot_num_data(self): return self.iterate.tot_num_data if self.iterate != NULL else None
    @tot_num_data.setter
    def tot_num_data(self, val): self.iterate.tot_num_data = val
    @property
    def modes(self): return self.iterate.modes if self.iterate != NULL else None
    @modes.setter
    def modes(self, val): self.iterate.modes = val
    
    @property
    def model1(self): return np.asarray(<double[:self.size**3]> self.iterate.model1).reshape(3*(self.size,)) if self.iterate != NULL else None
    @property
    def model2(self): return np.asarray(<double[:self.size**3]> self.iterate.model2).reshape(3*(self.size,)) if self.iterate != NULL else None
    @property
    def inter_weight(self): return np.asarray(<double[:self.size**3]> self.iterate.inter_weight).reshape(3*(self.size,)) if self.iterate != NULL else None
    @property
    def scale(self): return np.asarray(<double[:self.tot_num_data]> self.iterate.scale) if self.iterate != NULL else None
    @property
    def num_rel_quat(self): return np.asarray(<int[:self.tot_num_data]> self.iterate.num_rel_quat) if self.iterate != NULL else None
    @property
    def rel_quat(self):
        if self.iterate == NULL:
            return
        nrq = self.num_rel_quat
        flat = np.empty(nrq.sum(), dtype='i4')
        accum = np.insert(np.cumsum(nrq), 0, 0)
        for i in range(len(nrq)):
            flat[accum[i]:accum[i+1]] = np.asarray(<int[:nrq[i]]> self.iterate.rel_quat[i])
        return sparse.csr_matrix((np.ones(int(accum[-1])), flat, accum))
    #def rel_quat(self): return np.asarray(<double[:self.num_rel_quat.sum()]> self.iterate.rel_quat) if self.iterate != NULL else None
 
