import numpy as np
cimport numpy as np
cimport emc 
cimport openmp
from detector cimport detector
from dataset cimport dataset

cdef class dataset:
	def __init__(self):
		pass

	def parse_dataset(self, fname, detector det):
		cdef char* c_fname = fname
		cdef emc.detector c_det = det.det
		emc.parse_dataset(c_fname, &c_det, &self.dset)

	def parse_data(self, flist, detector det):
		cdef char* c_flist = flist
		cdef emc.detector c_det = det.det
		emc.parse_data(c_flist, &c_det, &self.dset)

	def make_blacklist(self, fname, int odd_flag=-1):
		cdef char* c_fname = fname
		emc.make_blacklist(c_fname, odd_flag, &self.dset)

	def calc_sum_fact(self, detector det):
		cdef emc.detector c_det = det.det
		emc.calc_sum_fact(&c_det, &self.dset)

	def free_data(self, scale_flag=False):
		cdef int c_scale_flag = int(scale_flag)
		emc.free_data(c_scale_flag, &self.dset)

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

	'''
	# Pointer to next dataset
	@property
	def next(self): 
		cdef object next_dset = self.dset.next[0]
		return next_dset
	'''
		
	# Need only be defined for head dataset
	@property
	def tot_num_data(self): return self.dset.tot_num_data
	@property
	def num_blacklist(self): return self.dset.num_blacklist
	@property
	def tot_mean_count(self): return self.dset.tot_mean_count


