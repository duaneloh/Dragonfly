from __future__ import print_function
import sys
import os
import numpy as np
import h5py

cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport FILE, fopen, fread, fclose
from libc.stdio cimport fseek, ftell, SEEK_END, SEEK_SET
from libc.string cimport memcpy, strcpy

from .detector cimport CDetector
from . cimport model as c_model
from .model cimport Model

cdef class Model:
    '''Model class for 3D/2D volume reconstruction.

    Args:
        size (int, optional): Size of the model grid. Default 0.
        num_modes (int, optional): Number of modes. Default 1.
        model_type (str, optional): Model type ('3d', '2d', or 'rz'). Default '3d'.
    '''

    def __init__(self, long size=0, int num_modes=1, model_type='3d'):
        '''Initialize Model object.'''
        self.mod = <c_model.model*> calloc(1, sizeof(c_model.model))
        if model_type.lower() == '3d':
            self.mod.mtype = MODEL_3D
        elif model_type.lower() == '2d':
            self.mod.mtype = MODEL_2D
        elif model_type.lower() == 'rz':
            self.mod.mtype = MODEL_RZ
        else:
            raise ValueError('Unknown model_type %s'%model_type)

        if size > 0:
            self.size = size
        self.num_modes = num_modes

    def allocate(self, fname, double model_mean=1., int rank=0):
        '''Allocate and optionally load model from file.

        Args:
            fname (str): Path to model file or empty string for random initialization.
            model_mean (float, optional): Mean value for random initialization. Default 1.0.
            rank (int, optional): MPI rank for parallel loading. Default 0.
        '''
        cdef FILE* fp
        cdef char* c_fname
        cdef long tot_vol
        cdef double[:] randvol

        ndim = 3 if self.mod.mtype == MODEL_3D else 2
        mshape = (self.num_modes,) + ndim*(self.size,)
        tot_vol = np.array(mshape).prod()

        self.mod.model1 = <double*> malloc(tot_vol * sizeof(double))
        self.mod.model2 = <double*> calloc(tot_vol, sizeof(double))
        self.mod.inter_weight = <double*> calloc(tot_vol, sizeof(double))

        if os.path.isfile(fname):
            print('Parsing model1 from', fname)
            if h5py.is_hdf5(fname):
                with h5py.File(fname, 'r') as f:
                    fshape = f['intens'].shape
                    dset = h5py.h5d.open(f.id, b'intens')
                    if fshape == mshape:
                        dset.read(h5py.h5s.ALL, h5py.h5s.ALL, self.model1)
                    elif fshape[1:] != mshape[1:]:
                        raise ValueError('Input model has wrong size (%d)' % fshape[1], fshape[1])
                    elif fshape[0] > mshape[0]:
                        print('More modes in file than expected, only reading in first', self.num_modes)
                        dspace = h5py.h5s.create_simple(mshape)
                        dset.read(h5py.h5s.ALL, dspace, self.model1)
                    else:
                        print('Not enough modes in file. Filling rest with white noise')
                        mspace = h5py.h5s.create_simple(fshape)
                        dset.read(mspace, h5py.h5s.ALL, self.model1)
                        randvol = np.random.random((mshape[0]-fshape[0],) + ndim*(self.size,)).ravel() * model_mean
                        memcpy(&self.mod.model1[fshape[0]*self.vol], &randvol[0], randvol.size*sizeof(double))
            else:
                c_fname = <char*> malloc(len(fname)+1)
                strcpy(c_fname, bytes(fname, 'utf-8'))
                fp = fopen(c_fname, 'rb')
                fseek(fp, 0, SEEK_END)
                fsize = int(np.rint((ftell(fp)/8.)**(1/3.)))
                if fsize != self.size:
                    raise ValueError('Wrong volume size in binary file (%d)' % fsize, fsize)
                fseek(fp, 0, SEEK_SET)
                fread(self.mod.model1, sizeof(double), tot_vol, fp)
                fclose(fp)
        else:
            print('Random model')
            randvol = np.random.random(mshape).ravel() * model_mean
            memcpy(self.mod.model1, &randvol[0], randvol.size*sizeof(double))

    def slice_gen(self, double[:] quat, CDetector det, int mode=0, view=None):
        '''Generate a 2D slice from the model.

        Args:
            quat (ndarray): Quaternion defining the orientation.
            det (CDetector): Detector to project onto.
            mode (int, optional): Mode index. Default 0.
            view (ndarray, optional): Pre-allocated output array.

        Returns:
            ndarray: 2D slice values.
        '''
        if self.mod.model1 == NULL:
            raise AttributeError('Allocate model1 first')

        cdef double[:] out_view
        if view is None:
            out_view = np.empty(det.num_pix, dtype='f8')
        else:
            out_view = view

        if self.mod.mtype == MODEL_3D:
            c_model.slice_gen3d(&quat[0], mode, &out_view[0], det.det, self.mod)
        elif self.mod.mtype == MODEL_2D:
            c_model.slice_gen2d(&quat[0], mode, &out_view[0], det.det, self.mod)
        elif self.mod.mtype == MODEL_RZ:
            c_model.slice_genrz(&quat[0], mode, &out_view[0], det.det, self.mod)
        return np.asarray(out_view)

    def slice_merge(self, double[:] quat, double[:] view, CDetector det, int mode=0):
        '''Merge a 2D slice back into the model.

        Args:
            quat (ndarray): Quaternion defining the orientation.
            view (ndarray): 2D slice values to merge.
            det (CDetector): Detector for projection.
            mode (int, optional): Mode index. Default 0.
        '''
        if self.mod.model2 == NULL:
            raise AttributeError('Allocate model2 first')

        if self.mod.mtype == MODEL_3D:
            c_model.slice_merge3d(&quat[0], mode, &view[0], &self.mod.model2[0], &self.mod.inter_weight[0], self.size, det.det)
        elif self.mod.mtype == MODEL_2D:
            c_model.slice_merge2d(&quat[0], mode, &view[0], &self.mod.model2[0], &self.mod.inter_weight[0], self.size, det.det)
        elif self.mod.mtype == MODEL_RZ:
            c_model.slice_mergerz(&quat[0], mode, &view[0], &self.mod.model2[0], &self.mod.inter_weight[0], self.size, det.det)

    def free(self):
        '''Free allocated model memory.'''
        if self.mod.model1 != NULL:
            free(self.mod.model1)
        if self.mod.model2 != NULL:
            free(self.mod.model2)
        if self.mod.inter_weight != NULL:
            free(self.mod.inter_weight)

    @staticmethod
    def symmetrize_friedel(double[:,:,:] model, double[:,:,:] weights):
        '''Apply Friedel (point inversion) symmetry.

        Args:
            model (ndarray): Model array to symmetrize.
            weights (ndarray): Weight array.
        '''
        cdef int size = model.shape[0]
        with nogil:
            c_model.symmetrize_friedel(&model[0,0,0], &weights[0,0,0], size)

    @staticmethod
    def symmetrize_octahedral(double[:,:,:] model, double[:,:,:] weights):
        '''Apply octahedral point group symmetry.

        Args:
            model (ndarray): Model array to symmetrize.
            weights (ndarray): Weight array.
        '''
        cdef int size = model.shape[0]
        with nogil:
            c_model.symmetrize_octahedral(&model[0,0,0], &weights[0,0,0], size)

    @staticmethod
    def symmetrize_icosahedral(double[:,:,:] model, double[:,:,:] weights):
        '''Apply icosahedral point group symmetry.

        Args:
            model (ndarray): Model array to symmetrize.
            weights (ndarray): Weight array.
        '''
        cdef int size = model.shape[0]
        with nogil:
            c_model.symmetrize_icosahedral(&model[0,0,0], &weights[0,0,0], size)

    @staticmethod
    def symmetrize_axial(double[:,:,:] model, double[:,:,:] weights, int order):
        '''Apply N-fold axial symmetry.

        Args:
            model (ndarray): Model array to symmetrize.
            weights (ndarray): Weight array.
            order (int): Order of rotational symmetry.
        '''
        cdef int size = model.shape[0]
        with nogil:
            c_model.symmetrize_axial(&model[0,0,0], &weights[0,0,0], size, order)

    @staticmethod
    def symmetrize_friedel2d(double[:,:,:] model2d, double[:,:,:] weights2d):
        '''Apply Friedel symmetry to 2D slices.

        Args:
            model2d (ndarray): 2D model array.
            weights2d (ndarray): 2D weight array.
        '''
        cdef int num_modes = model2d.shape[0]
        cdef int size = model2d.shape[1]
        with nogil:
            c_model.symmetrize_friedel2d(&model2d[0,0,0], &weights2d[0,0,0], num_modes, size)

    @staticmethod
    def rotate_model(double[:,:,:] model, double[:,:] rot, int max_r=0, rotmodel=None):
        '''Apply rotation matrix to model.

        Args:
            model (ndarray): Model array to rotate.
            rot (ndarray): 3x3 rotation matrix.
            max_r (int, optional): Maximum radius for masking.
            rotmodel (ndarray, optional): Pre-allocated output array.

        Returns:
            ndarray: Rotated model.
        '''
        cdef int i, j, size = model.shape[0]
        cdef double[:,:,:] rotmodel_view
        cdef double c_rot[3][3]
        for i in range(3):
            for j in range(3):
                c_rot[i][j] = rot[i,j]

        if rotmodel is None:
            rotmodel_view = np.empty(3*(size,))
        else:
            rotmodel_view = rotmodel

        with nogil:
            c_model.rotate_model(c_rot, &model[0,0,0], size, max_r, &rotmodel_view[0,0,0])
        return np.asarray(rotmodel_view)

    @staticmethod
    def make_rot_quat(double[:] quaternion):
        '''Convert quaternion to rotation matrix.

        Args:
            quaternion (ndarray): 4-component quaternion.

        Returns:
            ndarray: 3x3 rotation matrix.
        '''
        cdef double rot[3][3]
        c_model.make_rot_quat(&quaternion[0], rot)
        return np.asarray(rot)

    @property
    def mtype(self):
        '''Model type: 'MODEL_3D', 'MODEL_2D', or 'MODEL_RZ'.'''
        return ['MODEL_3D', 'MODEL_2D', 'MODEL_RZ'][self.mod.mtype]
    @property
    def ndim(self):
        '''Number of dimensions (3 for 3D, 2 for 2D/RZ).'''
        return 3 if self.mod.mtype == MODEL_3D else 2
    @property
    def size(self):
        '''Size of the model grid.'''
        return self.mod.size
    @size.setter
    def size(self, long val):
        '''Set size and update center and volume.'''
        self.mod.size = val
        self.mod.center = val // 2
        self.mod.vol = val**self.ndim
    @property
    def center(self):
        '''Center coordinate of the model.'''
        return self.mod.center
    @property
    def vol(self):
        '''Total volume (number of voxels).'''
        return self.mod.vol
    @property
    def num_modes(self):
        '''Number of modes.'''
        return self.mod.num_modes
    @num_modes.setter
    def num_modes(self, int val):
        '''Set number of modes.'''
        self.mod.num_modes = val

    @property
    def model1(self):
        '''Current model intensities, shape (num_modes, size, ...).'''
        return np.asarray(<double[:self.num_modes*self.size**self.ndim]>self.mod.model1).reshape((self.num_modes,) + self.ndim*(self.size,))
    @property
    def model2(self):
        '''Accumulated model from current iteration, shape (num_modes, size, ...).'''
        return np.asarray(<double[:self.num_modes*self.size**self.ndim]>self.mod.model2).reshape((self.num_modes,) + self.ndim*(self.size,))
    @property
    def inter_weight(self):
        '''Interpolation weights, shape (num_modes, size, ...).'''
        return np.asarray(<double[:self.num_modes*self.size**self.ndim]>self.mod.inter_weight).reshape((self.num_modes,) + self.ndim*(self.size,))
