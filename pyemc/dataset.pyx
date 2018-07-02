import numpy as np
#from builtins import bytes

cimport numpy as np
from libc.stdint cimport uint8_t
from libc.stdlib cimport malloc, free

cimport decl
from detector cimport detector
from dataset cimport dataset

cdef class dataset:
	def __init__(self, det, allocate=True):
		self.det = <detector> det
		if allocate:
			self._alloc()
		else:
			self.dset = NULL

	def _alloc(self):
		self.dset = <decl.dataset*> malloc(sizeof(decl.dataset))
		self.dset.num_data_prev = 0
		self.dset.next = NULL
		self.dset.blacklist = NULL
		self.dset.sum_fact = NULL

	def generate_data(self, config_fname, type_string=b'in', config_section=b'emc'):
		cdef char* c_config_fname = config_fname
		cdef char* c_config_section = config_section
		cdef char* c_type_string = type_string
		ret = decl.generate_data(c_config_fname, c_config_section, c_type_string, self.det.det, self.dset)
		assert ret == 0

	def calc_sum_fact(self):
		decl.calc_sum_fact(self.det.det, self.dset)

	def parse_dataset(self, fname):
		cdef char* c_fname = fname
		ret = decl.parse_dataset(c_fname, self.det.det, self.dset)
		assert ret == 0

	def parse_data(self, flist):
		cdef char* c_flist = flist
		ret = decl.parse_data(c_flist, self.det.det, self.dset)
		assert ret > 0
		return ret

	def generate_blacklist(self, config_fname):
		cdef char* c_config_fname = config_fname
		decl.generate_blacklist(c_config_fname, self.dset)

	def make_blacklist(self, fname, int odd_flag=-1):
		cdef char* c_fname = fname
		decl.make_blacklist(c_fname, odd_flag, self.dset)

	def free_data(self, scale_flag=False):
		decl.free_data(int(scale_flag), self.dset)
		self.dset = NULL

	def __del__(self):
		self.free_data()

	@property
	def type(self): return self.dset.type if self.dset != NULL else None
	@property
	def num_data(self): return self.dset.num_data if self.dset != NULL else None
	@property
	def num_pix(self): return self.dset.num_pix if self.dset != NULL else None
	@property
	def mean_count(self): return self.dset.mean_count if self.dset != NULL else None
	@property
	def filename(self): return <bytes>self.dset.filename if self.dset != NULL else None

	# Sparse dataset
	@property
	def ones_total(self): return self.dset.ones_total if self.dset != NULL else None
	@property
	def multi_total(self): return self.dset.multi_total if self.dset != NULL else None
	@property
	def ones(self): return np.asarray(<int[:self.num_data]>self.dset.ones, dtype='i4') if self.dset != NULL else None
	@property
	def multi(self): return np.asarray(<int[:self.num_data]>self.dset.multi, dtype='i4') if self.dset != NULL else None
	@property
	def place_ones(self): return np.asarray(<int[:self.ones_total]>self.dset.place_ones, dtype='i4') if self.dset != NULL else None
	@property
	def place_multi(self): return np.asarray(<int[:self.multi_total]>self.dset.place_multi, dtype='i4') if self.dset != NULL else None
	@property
	def count_multi(self): return np.asarray(<int[:self.multi_total]>self.dset.count_multi, dtype='i4') if self.dset != NULL else None
	@property
	def ones_accum(self): return np.asarray(<long[:self.num_data]>self.dset.ones_accum, dtype='i8') if self.dset != NULL else None
	@property
	def multi_accum(self): return np.asarray(<long[:self.num_data]>self.dset.multi_accum, dtype='i8') if self.dset != NULL else None

	# Dense dataset
	@property
	def frames(self): return np.asarray(<double[:self.num_data*self.num_pix]>self.dset.frames, dtype='f8').reshape(self.num_data, self.num_pix) if self.dset != NULL else None
	@property
	def int_frames(self): return np.asarray(<int[:self.num_data*self.num_pix]>self.dset.int_frames, dtype='i4').reshape(self.num_data, self.num_pix) if self.dset != NULL else None

	# Pointer to next dataset
	@property
	def next(self): 
		if self.dset == NULL:
			return None
		else:
			next_dset = dataset(self.det)
			next_dset.dset = self.dset.next
			return next_dset
		
	# Need only be defined for head dataset
	@property
	def tot_num_data(self): return self.dset.tot_num_data if self.dset != NULL else None
	@property
	def num_blacklist(self): return self.dset.num_blacklist if self.dset != NULL else None
	@property
	def tot_mean_count(self): return self.dset.tot_mean_count if self.dset != NULL else None
	@property
	def count(self): return np.asarray(<int[:self.tot_num_data]>self.dset.count, dtype='i4') if self.dset != NULL else None
	@property
	def sum_fact(self): return np.asarray(<double[:self.tot_num_data]>self.dset.sum_fact, dtype='f8') if self.dset != NULL else None
	@property
	def blacklist(self): return np.asarray(<uint8_t[:self.tot_num_data]>self.dset.blacklist, dtype='u1') if self.dset != NULL else None


