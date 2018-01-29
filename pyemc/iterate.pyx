import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport emc
from dataset cimport dataset
from detector cimport detector
from iterate cimport iterate
from params cimport params

cdef class iterate:
	def __init__(self):
		self.iterate = <emc.iterate*> PyMem_Malloc(sizeof(emc.iterate))
		self.iterate.size = -1
		self.iterate.model1 = NULL
		self.iterate.model2 = NULL
		self.iterate.inter_weight = NULL
		self.iterate.scale = NULL

	def generate_iterate(self, config_fname, double qmax, params param, detector det, dataset dset, continue_flag=False, config_section='emc'):
		cdef char* c_config_fname = config_fname
		cdef char* c_config_section = config_section
		ret = emc.generate_iterate(c_config_fname, c_config_section, int(continue_flag), qmax, param.param, det.det, dset.dset, self.iterate)
		assert ret == 0

	def calculate_size(self, double qmax):
		emc.calculate_size(qmax, self.iterate)

	def parse_scale(self, fname, dataset dset):
		cdef char* c_fname = fname
		return emc.parse_scale(c_fname, dset.dset, self.iterate)

	def calc_scale(self, dataset dset, detector det, print_fname=None):
		cdef char* c_print_fname
		if print_fname is not None:
			c_print_fname = print_fname
		else:
			c_print_fname = NULL
		emc.calc_scale(dset.dset, det.det, c_print_fname, self.iterate)

	def normalize_scale(self, dataset dset):
		emc.normalize_scale(dset.dset, self.iterate)

	def parse_input(self, fname, double mean, int rank=0, print_fname=None):
		cdef char* c_fname = fname
		cdef char* c_print_fname
		if print_fname is not None:
			c_print_fname = print_fname
		else:
			c_print_fname = NULL
		emc.parse_input(c_fname, mean, c_print_fname, rank, self.iterate)

	def free_iterate(self):
		emc.free_iterate(self.iterate)

	@property
	def size(self): return self.iterate.size if self.iterate != NULL else None
	@property
	def center(self): return self.iterate.center if self.iterate != NULL else None
	@property
	def model1(self): return np.asarray(<double[:self.size*self.size*self.size]> self.iterate.model1).reshape(self.size,self.size,self.size) if self.iterate != NULL else None
	@property
	def model2(self): return np.asarray(<double[:self.size*self.size*self.size]> self.iterate.model2).reshape(self.size,self.size,self.size) if self.iterate != NULL else None
	@property
	def inter_weight(self): return np.asarray(<double[:self.size*self.size*self.size]> self.iterate.inter_weight).reshape(self.size,self.size,self.size) if self.iterate != NULL else None
	def scale(self, dataset dset): return np.asarray(<double[:dset.tot_num_data]> self.iterate.scale) if self.iterate != NULL else None
