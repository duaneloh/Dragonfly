import numpy as np
cimport numpy as np
cimport decl 
cimport openmp
from detector cimport detector

def make_rot_quat(np.ndarray[double, ndim=1] quat):
    '''
    Returns rotation matrix corresponding to given quaternion
    
    make_rot_quat(np.ndarray[double, ndim=1] quat)
    '''
    cdef double[:,:] rotarr = np.zeros((3,3), dtype='f8')
    decl.make_rot_quat(&quat[0], <double(*)[3]>&rotarr[0][0])
    return np.asarray(rotarr).reshape(3,3)

def slice_gen3d(np.ndarray[double, ndim=1, mode='c'] quat,
                np.ndarray[double, ndim=1, mode='c'] out_slice,
                np.ndarray[double, ndim=3, mode='c'] model,
                detector det,
                double rescale=0):
    '''
    Interpolates values at given voxel positions from a given 3D volume for a given quaternion
    
    slice_gen3d(np.ndarray[double, ndim=1, mode='c'] quat,
                np.ndarray[double, ndim=1, mode='c'] out_slice,
                np.ndarray[double, ndim=3, mode='c'] model,
                detector det,
                double rescale=0)
    '''
    cdef int size = model.shape[0]
    cdef double[:,:,:] modelarr = model
    decl.slice_gen3d(&quat[0], rescale, &out_slice[0], <double*>&modelarr[0][0][0], size, det.det)

def slice_merge3d(np.ndarray[double, ndim=1, mode='c'] quat,
                  np.ndarray[double, ndim=1, mode='c'] in_slice,
                  np.ndarray[double, ndim=3, mode='c'] model,
                  np.ndarray[double, ndim=3, mode='c'] weight,
                  detector det):
    '''
    Merges slice into 3D model and interpolation weight arrays
    
    slice_merge3d(np.ndarray[double, ndim=1, mode='c'] quat,
                  np.ndarray[double, ndim=1, mode='c'] in_slice,
                  np.ndarray[double, ndim=3, mode='c'] model,
                  np.ndarray[double, ndim=3, mode='c'] weight,
                  detector det)
    '''
    cdef int size = model.shape[0]
    cdef double[:,:,:] modelarr = model
    cdef double[:,:,:] weightarr = weight
    decl.slice_merge3d(&quat[0], &in_slice[0], <double*>&modelarr[0][0][0], <double*>&weightarr[0][0][0], size, det.det)

def slice_gen2d(np.ndarray[double, ndim=1, mode='c'] angle,
                np.ndarray[double, ndim=1, mode='c'] out_slice,
                np.ndarray[double, ndim=3, mode='c'] model,
                detector det,
                double rescale=0):
    '''
    Interpolates values at given voxel positions from a list of 2D slices for a given angle pointer
    If there are N slices the angle has a range of [0,2 N Pi)
    
    slice_gen2d(np.ndarray[double, ndim=1, mode='c'] angle,
                np.ndarray[double, ndim=1, mode='c'] out_slice,
                np.ndarray[double, ndim=3, mode='c'] model,
                detector det,
                double rescale=0)
    '''
    cdef int size = model.shape[1]
    cdef double[:,:,:] modelarr = model
    decl.slice_gen2d(&angle[0], rescale, &out_slice[0], <double*>&modelarr[0][0][0], size, det.det)

def slice_merge2d(np.ndarray[double, ndim=1, mode='c'] angle,
                  np.ndarray[double, ndim=1, mode='c'] in_slice,
                  np.ndarray[double, ndim=3, mode='c'] model,
                  np.ndarray[double, ndim=3, mode='c'] weight,
                  detector det):
    '''
    Merges slice into slice stack and interpolation weight arrays using given angle
    If there are N slices the angle has a range of [0,2 N Pi)
    
    slice_merge2d(np.ndarray[double, ndim=1, mode='c'] angle,
                  np.ndarray[double, ndim=1, mode='c'] in_slice,
                  np.ndarray[double, ndim=3, mode='c'] model,
                  np.ndarray[double, ndim=3, mode='c'] weight,
                  detector det)
    '''
    cdef int size = model.shape[1]
    cdef double[:,:,:] modelarr = model
    cdef double[:,:,:] weightarr = weight
    decl.slice_merge2d(&angle[0], &in_slice[0], <double*>&modelarr[0][0][0], <double*>&weightarr[0][0][0], size, det.det)

def symmetrize_friedel(np.ndarray[np.double_t, ndim=3, mode='c'] model):
    '''
    Friedel symmetrizes 3D array
    
    symmetrize_friedel(np.ndarray[np.double_t, ndim=3, mode='c'] model)
    '''
    cdef int size = model.shape[0]
    cdef double[:,:,:] modelarr = model
    decl.symmetrize_friedel(<double*>&modelarr[0][0][0], size)

def symmetrize_icosahedral(np.ndarray[np.double_t, ndim=3, mode='c'] model):
    '''
    Icosahedrally symmetrizes 3D array
    
    symmetrize_friedel(np.ndarray[np.double_t, ndim=3, mode='c'] model)
    '''
    cdef int size = model.shape[0]
    cdef double[:,:,:] modelarr = model
    decl.symmetrize_icosahedral(<double*>&modelarr[0][0][0], size)

def rotate_model(np.ndarray[double, ndim=2] rot,
                 np.ndarray[double, ndim=3, mode='c'] model,
                 np.ndarray[double, ndim=3, mode='c'] rotmodel):
    '''
    Rotates 3D cubic array by given rotation matrix and write to rotmodel
    
    rotate_model(np.ndarray[double, ndim=2] rot,
                 np.ndarray[double, ndim=3, mode='c'] model,
                 np.ndarray[double, ndim=3, mode='c'] rotmodel)
    '''
    cdef int size = model.shape[0]
    cdef double[:,:] rotarr = rot
    cdef double[:,:,:] modelarr = model
    cdef double[:,:,:] rotmodelarr = rotmodel 
    decl.rotate_model(<double(*)[3]>&rotarr[0][0], <double*>&modelarr[0,0,0], size, <double*>&rotmodelarr[0,0,0])

