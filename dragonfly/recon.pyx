import numpy as np
from mpi4py import MPI

cimport numpy as np
cimport openmp
from libc.stdlib cimport malloc, calloc, free
from .iterate cimport Iterate
from . cimport recon as c_recon
from . cimport model as c_model
from . cimport params as c_params

cdef class EMCRecon():
    def __init__(self, num_threads=-1):
        if num_threads <= 0:
            self.num_threads = openmp.omp_get_max_threads()
        else:
            self.num_threads = num_threads
        openmp.omp_set_num_threads(self.num_threads)

        self.mdata = <c_recon.max_data*> calloc(1, sizeof(c_recon.max_data))
        self.mdata.within_openmp = 0

    def set_iterate(self, Iterate itr):
        if itr.iter.dset == NULL:
            print('Set data for iterate first')
            return
        if itr.iter.mod == NULL:
            print('Set model for iterate first')
            return
        if itr.iter.quat == NULL:
            print('Set quat for iterate first')
            return
        if itr.iter.par == NULL:
            print('Set params for iterate first')
            return

        itr.iter.quat.num_rot_p = itr.quat.divide(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)
        itr.iter.par.rank = MPI.COMM_WORLD.rank
        itr.iter.par.num_proc = MPI.COMM_WORLD.size
        self.mdata.iter = itr.iter

    def run_iteration(self):
        cdef double likelihood, beta_mean
        
        if self.mdata.iter == NULL:
            print('Set iterate first')
            return

        if self.mdata.iter.par.rtype == c_params.RECON3D:
            c_recon.slice_gen = &c_model.slice_gen3d
            c_recon.slice_merge = &c_model.slice_merge3d

        #if self.iter.par.iteration == 1:
        #    self.write_log_file_header()

        cdef long vol = self.mdata.iter.mod.vol
        MPI.COMM_WORLD.Bcast([<double[:vol]>self.mdata.iter.mod.model1, MPI.DOUBLE], 0)
        beta_mean = self.update_beta()
        # self.update_radius()
        likelihood = c_recon.maximize(self.mdata)

        #if self.mdata.iter.par.rank == 0:
        #    self.update_model(likelihood)
        if self.mdata.iter.par.need_scaling == 1 and self.mdata.iter.mod.mtype == c_model.MODEL_3D:
            self.iter.normalize_scale()
        #if self.mdata.iter.par.rank == 0:
        #    self.save_models()
        #    self.update_log_file()

        if self.mdata.iter.par.rank == 0:
            print('Finished maximize')

    def update_beta(self):
        cdef int d
        cdef double factor, beta_mean = 0.

        if self.mdata.iter.par.beta_factor <= 0.:
            factor = self.mdata.iter.par.beta_jump ** ((self.mdata.iter.par.iteration-1) // self.mdata.iter.par.beta_period)
        else:
            factor = self.mdata.iter.par.beta_factor

        for d in range(self.mdata.iter.tot_num_data):
            if self.mdata.iter.blacklist[d] == 0:
                self.mdata.iter.beta[d] = self.mdata.iter.beta_start[d] * factor
                if self.mdata.iter.beta[d] > 1.:
                    self.mdata.iter.beta[d] = 1.
                beta_mean += self.mdata.iter.beta[d]
        beta_mean /= (self.mdata.iter.tot_num_data - self.mdata.iter.num_blacklist)

        return beta_mean

    def free(self):
        if self.mdata == NULL or self.mdata.iter == NULL:
            return
        c_recon.free_max_data(self.mdata)

    @property
    def num_threads(self): return self.num_threads
    @num_threads.setter
    def num_threads(self, int val): self.num_threads = val

    @property
    def iter(self):
        if self.mdata.iter == NULL:
            return
        retval = Iterate()
        retval.free()
        retval.iter = self.mdata.iter
        return retval

    # Flags
    @property
    def refinement(self): return self.mdata.refinement
    @refinement.setter
    def refinement(self, int val): self.mdata.refinement = val
    @property
    def within_openmp(self): return self.mdata.within_openmp
    @within_openmp.setter
    def within_openmp(self, int val): self.mdata.within_openmp = val


    # Private to OpenMP thread only
    @property
    def model(self): return np.asarray(<double[:self.mdata.iter.mod.vol]>self.mdata.model).reshape(3*(self.mdata.iter.mod.size,)) if self.mdata.model != NULL else None
    @property
    def weight(self): return np.asarray(<double[:self.mdata.iter.mod.vol]>self.mdata.weight).reshape(3*(self.mdata.iter.mod.size,)) if self.mdata.weight != NULL else None
    @property
    def all_views(self):
        cdef int num_pix
        if self.mdata.all_views == NULL:
            return
        else:
            retval = []
            for i in range(self.mdata.iter.num_det):
                num_pix = self.mdata.iter.det[i].num_pix
                retval.append(<double[:num_pix]> self.mdata.all_views[i])
            return retval
    @property
    def psum_r(self): return np.asarray(<double[:self.mdata.iter.quat.num_rot_p]>self.mdata.psum_r) if self.mdata.psum_r != NULL else None
    @property
    def psum_d(self): return np.asarray(<double[:self.mdata.iter.tot_num_data]>self.mdata.psum_d) if self.mdata.psum_d != NULL else None

    @property
    def max_exp(self): return np.asarray(<double[:self.mdata.iter.tot_num_data]>self.mdata.max_exp) if self.mdata.max_exp != NULL else None
    @property
    def p_norm(self): return np.asarray(<double[:self.mdata.iter.tot_num_data]>self.mdata.p_norm) if self.mdata.p_norm != NULL else None
    @property
    def u(self):
        if self.mdata.u == NULL:
            return
        retval = []
        for i in range(self.mdata.iter.num_det):
            retval.append(<double[:self.mdata.iter.quat.num_rot_p]> self.mdata.u[i])
        return retval
    @property
    def offset_prob(self): return np.asarray(<int[:self.num_threads*self.mdata.iter.tot_num_data]>self.mdata.offset_prob).reshape(self.num_threads, -1) if self.mdata.offset_prob != NULL else None

    @property
    def max_exp_p(self): return np.asarray(<double[:self.mdata.iter.tot_num_data]>self.mdata.max_exp_p) if self.mdata.max_exp_p != NULL else None
    @property
    def info(self): return np.asarray(<double[:self.mdata.iter.tot_num_data]>self.mdata.info) if self.mdata.info != NULL else None
    @property
    def likelihood(self): return np.asarray(<double[:self.mdata.iter.tot_num_data]>self.mdata.likelihood) if self.mdata.likelihood != NULL else None
    @property
    def rmax(self): return np.asarray(<int[:self.mdata.iter.tot_num_data]>self.mdata.rmax) if self.mdata.rmax != NULL else None
    @property
    def quat_norm(self): return np.asarray(<double[:self.mdata.iter.tot_num_data*self.mdata.iter.mod.num_modes]>self.mdata.quat_norm).reshape(-1, self.iter.mod.num_modes) if self.mdata.quat_norm != NULL else None
    @property
    def num_prob(self): return np.asarray(<int[:self.mdata.iter.tot_num_data]>self.mdata.num_prob) if self.mdata.num_prob != NULL else None
    @property
    def prob(self): return [np.asarray(<double[:self.mdata.num_prob[d]]>self.mdata.prob[d]) if self.mdata.num_prob[d] > 0 else None for d in range(self.mdata.iter.tot_num_data)] if self.mdata.prob != NULL else None
    @property
    def place_prob(self): return [np.asarray(<int[:self.mdata.num_prob[d]]>self.mdata.place_prob[d]) for d in range(self.mdata.iter.tot_num_data)] if self.mdata.place_prob != NULL else None

