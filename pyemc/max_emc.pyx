import os
import sys
import numpy as np
from scipy import sparse
from contextlib import contextmanager

from libc.stdlib cimport malloc, free
from libc.float cimport DBL_MAX
from libc.stdint cimport uint8_t
from posix.time cimport gettimeofday 
cimport numpy as np

cimport decl
cimport emc
cimport max_emc

@contextmanager
def _silence_stderr(to=os.devnull):
    fdes = sys.stderr.fileno()
    
    def _redirect_to(to):
        sys.stderr.close()
        os.dup2(to.fileno(), fdes)
        sys.stderr = os.fdopen(fdes, 'w')
    
    with os.fdopen(os.dup(fdes), 'w') as old_stderr:
        with open(to, 'w') as fptr:
            _redirect_to(fptr)
        try:
            yield
        finally:
            _redirect_to(old_stderr)

cdef class py_max_data:
    def __init__(self, refinement=0, within_openmp=0):
        self.data = <emc.max_data*>malloc(sizeof(emc.max_data))
        
        self.data.within_openmp = int(within_openmp)
        self.data.refinement = int(refinement)
        
        # Common only
        self.data.max_exp = NULL
        self.data.u = NULL
        self.data.p_norm = NULL
        self.data.offset_prob = NULL
        
        # OpenMP Private only
        self.data.model = NULL
        self.data.weight = NULL
        self.data.all_views = NULL
        self.data.psum_r = NULL
        self.data.psum_d = NULL
        self.data.mask = NULL
        
        # Both
        self.data.max_exp_p = NULL
        self.data.info = NULL
        self.data.likelihood = NULL
        self.data.rmax = NULL
        self.data.quat_norm = NULL
        self.data.prob = NULL
        self.data.num_prob = NULL
        self.data.place_prob = NULL

        # Background-scaling update (private only)
        self.data.G_old = NULL
        self.data.G_new = NULL
        self.data.G_latest = NULL
        self.data.W_old = NULL
        self.data.W_new = NULL
        self.data.W_latest = NULL
        self.data.scale_old = NULL
        self.data.scale_new = NULL
        self.data.scale_latest = NULL

    @property
    def within_openmp(self):
        return self.data.within_openmp
    @within_openmp.setter
    def within_openmp(self, value):
        self.data.within_openmp = value

    @property
    def model(self):
        return np.asarray(<double[:emc.iter.size**3]> self.data.model).reshape(3*(emc.iter.size,)) if self.data.model != NULL else None
    @property
    def weight(self):
        return np.asarray(<double[:emc.iter.size**3]> self.data.weight).reshape(3*(emc.iter.size,)) if self.data.weight != NULL else None
    @property
    def all_views(self):
        if self.data.all_views == NULL:
            return None
        else:
            retval = []
            for i in range(emc.det.num_det):
                retval.append(<double[:emc.det[i].num_pix]> self.data.all_views[i])
            return retval
    @property
    def psum_r(self):
        return np.asarray(<double[:emc.quat.num_rot_p]> self.data.psum_r) if self.data.psum_r != NULL else None
    @property
    def psum_d(self):
        return np.asarray(<double[:emc.frames.tot_num_data]> self.data.psum_d) if self.data.psum_d != NULL else None

    @property
    def max_exp(self):
        return np.asarray(<double[:emc.frames.tot_num_data]> self.data.max_exp) if self.data.max_exp != NULL else None
    @property
    def p_norm(self):
        return np.asarray(<double[:emc.frames.tot_num_data]> self.data.p_norm) if self.data.p_norm != NULL else None
    @property
    def u(self):
        if self.data.u == NULL:
            return None
        retval = []
        for i in range(emc.det.num_det):
            retval.append(<double[:emc.quat.num_rot_p]> self.data.u[i])
        return retval
    # TODO offset_prob[num_threads]

    @property
    def max_exp_p(self):
        return np.asarray(<double[:emc.frames.tot_num_data]> self.data.max_exp_p) if self.data.max_exp_p != NULL else None
    @property
    def info(self):
        return np.asarray(<double[:emc.frames.tot_num_data]> self.data.info) if self.data.info != NULL else None
    @property
    def likelihood(self):
        return np.asarray(<double[:emc.frames.tot_num_data]> self.data.likelihood) if self.data.likelihood != NULL else None
    @property
    def rmax(self):
        return np.asarray(<int[:emc.frames.tot_num_data]> self.data.rmax) if self.data.rmax != NULL else None
    @property
    def quat_norm(self):
        return np.asarray(<double[:emc.param.modes]> self.data.quat_norm) if self.data.quat_norm != NULL else None

    @property
    def num_prob(self):
        return np.asarray(<int[:emc.frames.tot_num_data]> self.data.num_prob) if self.data.num_prob != NULL else None
    @property
    def prob(self):
        if self.data.prob == NULL:
            return None
        nprob = self.num_prob
        if nprob.sum() == 0:
            return None
        flat = np.empty(nprob.sum(), dtype='i4')
        vals = np.empty(nprob.sum(), dtype='f8')
        accum = np.insert(np.cumsum(nprob), 0, 0)
        for i in range(len(nprob)):
            flat[accum[i]:accum[i+1]] = np.asarray(<int[:nprob[i]]> self.data.place_prob[i])
            vals[accum[i]:accum[i+1]] = np.asarray(<double[:nprob[i]]> self.data.prob[i])
        return sparse.csr_matrix((vals, flat, accum))

cdef class py_maximize:
    def __init__(self, config_fname, quiet_setup=True):
        gettimeofday(&tm1, NULL)
        if config_fname is not None:
            if quiet_setup:
                with _silence_stderr(): emc.setup(config_fname, 0)
            else:
                emc.setup(config_fname, 0)

    def allocate_memory(self, py_max_data data):
        max_emc.allocate_memory(data.data)

    def calculate_rescale(self, py_max_data common_data):
        max_emc.calculate_rescale(common_data.data)

    def calculate_prob(self, int r, py_max_data priv_data, py_max_data common_data):
        max_emc.calculate_prob(r, priv_data.data, common_data.data)

    def normalize_prob(self, py_max_data priv_data, py_max_data common_data):
        max_emc.normalize_prob(priv_data.data, common_data.data)

    def update_tomogram(self, int r, py_max_data priv_data, py_max_data common_data):
        max_emc.update_tomogram(r, priv_data.data, common_data.data)

    def merge_tomogram(self, int r, py_max_data priv_data):
        max_emc.merge_tomogram(r, priv_data.data)

    def combine_information_omp(self, py_max_data priv_data, py_max_data common_data):
        max_emc.combine_information_omp(priv_data.data, common_data.data)

    def combine_information_mpi(self, py_max_data common_data):
        avg_likelihood = max_emc.combine_information_mpi(common_data.data)
        return avg_likelihood

    def update_scale(self, py_max_data common_data):
        max_emc.update_scale(common_data.data)

    def free_memory(self, py_max_data data):
        max_emc.free_memory(data.data)
