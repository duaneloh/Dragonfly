import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport emc
from dataset cimport dataset
from detector cimport detector
from iterate cimport iterate

cdef class iterate:
	def __init__(self):
		self.iterate = <emc.iterate*> PyMem_Malloc(sizeof(emc.iterate))

	def parse_scale(self, fname, dataset dset):
		cdef char* c_fname = fname
		emc.parse_scale(c_fname, dset.dset, self.iterate)

	def calc_scale(self, dataset dset, detector det, print_fname=None):
		cdef char* c_print_fname
		if print_fname is not None:
			c_print_fname = print_fname
		else:
			c_print_fname = NULL
		emc.calc_scale(dset.dset, det.det, c_print_fname, self.iterate)

	def normalize_scale(self, dataset dset):
		emc.normalize_scale(dset.dset, self.iterate)

	def parse_input(self, fname, double mean, print_fname=None):
		cdef char* c_fname = fname
		cdef char* c_print_fname
		if print_fname is not None:
			c_print_fname = print_fname
		else:
			c_print_fname = NULL
		emc.parse_input(c_fname, mean, c_print_fname, self.iterate)

	def free_iterate(self, scale_flag=False):
		emc.free_iterate(int(scale_flag), self.iterate)
		PyMem_Free(self.iterate)

	@property
	def size(self): return self.iterate.size
	@property
	def center(self): return self.iterate.center
	@property
	def model1(self): return np.asarray(<double[:self.size*self.size*self.size]> self.iterate.model1).reshape(self.size,self.size,self.size)
	@property
	def model2(self): return np.asarray(<double[:self.size*self.size*self.size]> self.iterate.model2).reshape(self.size,self.size,self.size)
	@property
	def inter_weight(self): return np.asarray(<double[:self.size*self.size*self.size]> self.iterate.inter_weight).reshape(self.size,self.size,self.size)
	def scale(self, dataset dset): return np.asarray(<double[:dset.tot_num_data]> self.iterate.scale)
