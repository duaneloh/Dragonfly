import numpy as np
cimport numpy as np
cimport emc 
cimport openmp
from detector cimport detector

def make_rot_quat(np.ndarray[double, ndim=1] quat):
	'''Return rotation matric corresponding to given quaternion'''
	cdef double[:,:] rotarr = np.zeros(3,3)
	emc.make_rot_quat(&quat[0], <double(*)[3]>&rotarr[0][0])
	return np.asarray(rotarr).reshape(3,3)

def slice_gen(np.ndarray[double, ndim=1, mode='c'] quat,
              np.ndarray[double, ndim=1, mode='c'] out_slice,
              np.ndarray[double, ndim=3, mode='c'] model,
              detector det,
              double rescale=0):
	'''Interpolate vlues at given voxel positions from a given 3D volume for a given quaternion'''
	cdef int size = model.shape[0]
	cdef np.ndarray[double, mode='c'] flat_model = np.ascontiguousarray(model.flatten())
	emc.slice_gen(&quat[0], rescale, &out_slice[0], &flat_model[0], size, det.det)
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
	emc.slice_merge(&quat[0], &in_slice[0], &flat_model[0], &flat_weight[0], size, det.det)
	return flat_model.reshape(size,size,size), flat_weight.reshape(size,size,size)

def symmetrize_friedel(np.ndarray[np.double_t, ndim=3, mode='c'] model):
	'''Friedel symmetrize 3D array'''
	cdef int size = model.shape[0]
	cdef np.ndarray[double, mode='c'] flat_model = np.ascontiguousarray(model.flatten())
	emc.symmetrize_friedel(&flat_model[0], size)
	return flat_model.reshape(size,size,size)

def rotate_model(np.ndarray[double, ndim=2] rot,
                 np.ndarray[double, ndim=3, mode='c'] model,
				 np.ndarray[double, ndim=3, mode='c'] rotmodel):
	'''Rotate 3D cubic array by given rotation matrix and write to rotmodel'''
	cdef int size = model.shape[0]
	cdef double[:,:] rotarr = rot
	cdef np.ndarray[double, mode='c'] flat_model = np.ascontiguousarray(model.flatten())
	cdef np.ndarray[double, mode='c'] flat_rotmodel = np.ascontiguousarray(rotmodel.flatten())
	emc.rotate_model(<double(*)[3]>&rotarr[0][0], &model[0,0,0], size, &rotmodel[0,0,0])

