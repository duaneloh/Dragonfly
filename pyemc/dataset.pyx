import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport emc 
cimport openmp
from detector cimport detector
from dataset cimport dataset

cdef class dataset:
	def __init__(self):
		self.dset = <emc.dataset*> PyMem_Malloc(sizeof(emc.dataset))
		self.dset.next = NULL
		self.dset.blacklist = NULL
		self.dset.sum_fact = NULL

	def generate_data(self, config_fname, detector det, type_string='in', config_section='emc', rank=0, num_proc=1):
		if emc.config_section[0] == '\0':
			emc.config_section[:len(config_section)] = config_section
			emc.config_section[len(config_section)] = '\0'
		if emc.rank == 0: emc.rank = rank
		if emc.num_proc == 0: emc.num_proc = num_proc
		
		cdef char* c_config_fname = config_fname
		cdef char* c_type_string = type_string
		emc.generate_data(c_config_fname, c_type_string, det.det, self.dset)

	def parse_dataset(self, fname, detector det):
		cdef char* c_fname = fname
		emc.parse_dataset(c_fname, det.det, self.dset)

	def parse_data(self, flist, detector det):
		cdef char* c_flist = flist
		emc.parse_data(c_flist, det.det, self.dset)

	def make_blacklist(self, fname, int odd_flag=-1):
		cdef char* c_fname = fname
		emc.make_blacklist(c_fname, odd_flag, self.dset)

	def calc_sum_fact(self, detector det):
		emc.calc_sum_fact(det.det, self.dset)

	def free_data(self, scale_flag=False):
		cdef int c_scale_flag = int(scale_flag)
		emc.free_data(c_scale_flag, self.dset)

	@property
	def type(self): return self.dset.type
	@property
	def num_data(self): return self.dset.num_data
	@property
	def num_pix(self): return self.dset.num_pix
	@property
	def mean_count(self): return self.dset.mean_count
	@property
	def filename(self): return str(self.dset.filename)

	# Sparse dataset
	@property
	def ones_total(self): return self.dset.ones_total
	@property
	def multi_total(self): return self.dset.multi_total
	@property
	def ones(self): return np.asarray(<int[:self.num_data]>self.dset.ones, dtype='i4')
	@property
	def multi(self): return np.asarray(<int[:self.num_data]>self.dset.multi, dtype='i4')
	@property
	def place_ones(self): return np.asarray(<int[:self.ones_total]>self.dset.place_ones, dtype='i4')
	@property
	def place_multi(self): return np.asarray(<int[:self.multi_total]>self.dset.place_multi, dtype='i4')
	@property
	def count_multi(self): return np.asarray(<int[:self.multi_total]>self.dset.count_multi, dtype='i4')
	@property
	def ones_accum(self): return np.asarray(<long[:self.num_data]>self.dset.ones_accum, dtype='i8')
	@property
	def multi_accum(self): return np.asarray(<long[:self.num_data]>self.dset.multi_accum, dtype='i8')

	# Dense dataset
	@property
	def frames(self): return np.asarray(<double[:self.num_data*self.num_pix]>self.dset.frames, dtype='f8').reshape(self.num_data, self.num_pix)
	@property
	def int_frames(self): return np.asarray(<int[:self.num_data*self.num_pix]>self.dset.int_frames, dtype='i4').reshape(self.num_data, self.num_pix)

	# Pointer to next dataset
	@property
	def next(self): 
		next_dset = dataset()
		next_dset.dset = self.dset.next
		return next_dset
		
	# Need only be defined for head dataset
	@property
	def tot_num_data(self): return self.dset.tot_num_data
	@property
	def num_blacklist(self): return self.dset.num_blacklist
	@property
	def tot_mean_count(self): return self.dset.tot_mean_count


