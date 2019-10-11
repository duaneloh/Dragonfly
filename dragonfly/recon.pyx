cimport openmp
from .iterate cimport Iterate
from . cimport recon as c_recon
from . cimport model as c_model
from . cimport params as c_params

cdef class EMCRecon():
    def __init__(self, int num_threads=-1):
        if num_threads <= 0:
            openmp.omp_set_num_threads(openmp.omp_get_max_threads())
        else:
            openmp.omp_set_num_threads(num_threads)

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

        self.iter = itr.iter

    def run_iteration(self):
        if self.iter.par.rtype == c_params.RECON3D:
            c_recon.slice_gen = &c_model.slice_gen3d
            c_recon.slice_merge = &c_model.slice_merge3d
        c_recon.maximize(self.iter)
        print('Finished maximize')

    def update_beta(self):
        cdef int d
        cdef double factor, beta_mean = 0.

        if self.iter.par.beta_factor <= 0.:
            factor = self.iter.par.beta_jump ** ((self.iter.par.iteration-1) / self.iter.par.beta_period)
        else:
            factor = self.iter.par.beta_factor

        for d in range(self.iter.tot_num_data):
            if self.iter.blacklist[d] > 0:
                self.iter.beta[d] = self.iter.beta_start[d] * factor
                if self.iter.beta[d] > 1.:
                    self.iter.beta[d] = 1.
                beta_mean += self.iter.beta[d]
        beta_mean /= (self.iter.tot_num_data - self.iter.num_blacklist)

        return beta_mean
