import numpy as np
cimport numpy as np
cimport emc 
cimport openmp

cdef extern from "numpy/arrayobject.h":
	void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

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

