import numpy as np
cimport numpy as np
cimport emc 
cimport openmp

cdef extern from "numpy/arrayobject.h":
	void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

# detector.c class and functions
# ============================================================
cdef class detector:
	cdef emc.detector det

	def __init__(self, fname, norm_flag=True):
		cdef char* c_fname = fname
		emc.parse_detector(c_fname, &self.det, int(norm_flag))

	def __del__(self):
		emc.free_detector(&self.det)

	@property
	def num_pix(self): return self.det.num_pix
	@property
	def rel_num_pix(self): return self.det.rel_num_pix
	@property
	def detd(self): return self.det.detd
	@property
	def ewald_rad(self): return self.det.ewald_rad
	@property
	def pixels(self): return np.asarray(<double[:4*self.num_pix]>self.det.pixels).reshape(-1,4)
	@property
	def mask(self): return np.asarray(<emc.uint8_t[:self.num_pix]>self.det.mask)

	# Only relevant for first detector in list
	@property
	def num_det(self): return self.det.num_det
	@property
	def num_dfiles(self): return self.det.num_dfiles
	@property
	def mapping(self): return np.asarray(self.det.mapping, dtype='i4')

def generate_size(double qmax, np.ndarray[long] size, np.ndarray[long] center):
	'''
	Calculates size and center given a qmax
	Since these values must be replaced in-place, they must be numpy arrays
	'''
	cdef long [:] size_view = size
	#cdef long [:] center_view = center
	emc.generate_size(qmax, &size[0], &center[0])

# dataset.c class and functions
# ============================================================
cdef class dataset:
	cdef emc.dataset dset

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

# interp.c functions
# ============================================================
def slice_gen(np.ndarray[double, ndim=1, mode='c'] quat,
              np.ndarray[double, ndim=1, mode='c'] out_slice,
              np.ndarray[double, ndim=3, mode='c'] model,
              detector det,
	      double rescale=0):
	'''Interpolate vlues at given voxel positions from a given 3D volume for a given quaternion'''
	cdef int size = model.shape[0]
	cdef np.ndarray[double, mode='c'] flat_model = np.ascontiguousarray(model.flatten())
	emc.slice_gen(&quat[0], rescale, &out_slice[0], &flat_model[0], size, &det.det)
	return out_slice

def slice_merge(np.ndarray[double, ndim=1, mode='c'] quat,
              np.ndarray[double, ndim=1, mode='c'] in_slice,
              np.ndarray[double, ndim=3, mode='c'] model,
              np.ndarray[double, ndim=3, mode='c'] weight,
	      detector det):
	'''
	Merges slice into 3D model and interpolation weight arrays
	Returns tuple of updated 3D model and 3D weights
	'''
	cdef int size = model.shape[0]
	cdef np.ndarray[double, mode='c'] flat_model = np.ascontiguousarray(model.flatten())
	cdef np.ndarray[double, mode='c'] flat_weight = np.ascontiguousarray(weight.flatten())
	emc.slice_merge(&quat[0], &in_slice[0], &flat_model[0], &flat_weight[0], size, &det.det)
	return flat_model.reshape(size,size,size), flat_weight.reshape(size,size,size)

def friedel_sym(np.ndarray[np.double_t, ndim=3, mode='c'] model):
	'''Friedel symmetrize 3D array'''
	cdef int size = model.shape[0]
	cdef np.ndarray[double, mode='c'] flat_model = np.ascontiguousarray(model.flatten())
	emc.symmetrize_friedel(&flat_model[0], size)
	return flat_model.reshape(size,size,size)

