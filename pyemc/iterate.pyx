import numpy as np
cimport numpy as np

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

	def calc_scale(self, dataset dset, detector det, print_fname=None):
		cdef char* c_print_fname
		if print_fname is not None:
			c_print_fname = print_fname
		else:
			c_print_fname = NULL
		decl.calc_scale(dset.dset, det.det, c_print_fname, self.iterate)

	def normalize_scale(self, dataset dset):
		decl.normalize_scale(dset.dset, self.iterate)

	def parse_input(self, fname, double mean, int rank=0, print_fname=None):
		cdef char* c_fname = fname
		cdef char* c_print_fname
		if print_fname is not None:
			c_print_fname = print_fname
		else:
			c_print_fname = NULL
		decl.parse_input(c_fname, mean, c_print_fname, rank, 42, self.iterate)

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
	@property
	def modes(self): return self.iterate.modes if self.iterate != NULL else None
	@property
	def model1(self): return np.asarray(<double[:self.size**3]> self.iterate.model1).reshape(3*(self.size,)) if self.iterate != NULL else None
	@property
	def model2(self): return np.asarray(<double[:self.size**3]> self.iterate.model2).reshape(3*(self.size,)) if self.iterate != NULL else None
	@property
	def inter_weight(self): return np.asarray(<double[:self.size**3]> self.iterate.inter_weight).reshape(3*(self.size,)) if self.iterate != NULL else None
	@property
	def scale(self): return np.asarray(<double[:self.tot_num_data]> self.iterate.scale) if self.iterate != NULL else None
 
