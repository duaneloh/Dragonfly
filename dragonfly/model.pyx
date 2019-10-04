from __future__ import print_function
import sys
import os
import numpy as np
from scipy import ndimage
import h5py
import pandas

cimport numpy as np
from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport FILE, fopen, fread, fclose
from libc.string cimport memcpy, strcpy

from .detector cimport CDetector
from . cimport model as c_model
from .model cimport Model

cdef class Model:
    def __init__(self, long size=0, int num_modes=1, model_type='3d'):
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
        cdef FILE* fp
        cdef char* c_fname
        cdef long tot_vol
        cdef double[:] randvol

        ndim = 3 if self.mod.mtype == MODEL_3D else 2
        mshape = (self.num_modes,) + ndim*(self.size,)
        tot_vol = np.array(mshape).prod()

        self.mod.model1 = <double*> malloc(tot_vol * sizeof(double))
        self.mod.model2 = <double*> malloc(tot_vol * sizeof(double))
        self.mod.inter_weight = <double*> malloc(tot_vol * sizeof(double))

        if os.path.isfile(fname):
            print('Parsing model1 from', fname)
            if h5py.is_hdf5(fname):
                with h5py.File(fname, 'r') as f:
                    fshape = f['intens'].shape
                    dset = h5py.h5d.open(f.id, b'intens')
                    if fshape == mshape:
                        dset.read(h5py.h5s.ALL, h5py.h5s.ALL, self.model1)
                    elif fshape[1:] != mshape[1:]:
                        raise ValueError('Input model has wrong grid size')
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
                fread(self.mod.model1, sizeof(double), tot_vol, fp)
                fclose(fp)
        else:
            print('Random model')
            randvol = np.random.random(mshape).ravel() * model_mean
            memcpy(self.mod.model1, &randvol[0], randvol.size*sizeof(double))

    def slice_gen(self, double[:] quat, CDetector det, int mode=0, out_slice=None):
        if self.mod.model1 == NULL:
            raise AttributeError('Allocate model1 first')

        cdef double[:] out_slice_view
        if out_slice is None:
            out_slice_view = np.empty(det.num_pix, dtype='f8')
        else:
            out_slice_view = out_slice
        print(out_slice_view.shape)
        if self.mod.mtype == MODEL_3D:
            print('3D model')
            c_model.slice_gen3d(&quat[0], mode, &out_slice_view[0], det.det, self.mod)
        elif self.mod.mtype == MODEL_2D:
            print('2D model')
            c_model.slice_gen2d(&quat[0], mode, &out_slice_view[0], det.det, self.mod)
        elif self.mod.mtype == MODEL_RZ:
            c_model.slice_genrz(&quat[0], mode, &out_slice_view[0], det.det, self.mod)
        return np.asarray(out_slice_view)

    @staticmethod
    def make_rot_quat(double[:] quaternion):
        cdef double rot[3][3]
        c_model.make_rot_quat(&quaternion[0], rot)
        return np.asarray(rot)

    @property
    def mtype(self): return ['MODEL_3D', 'MODEL_2D', 'MODEL_RZ'][self.mod.mtype]
    @property
    def ndim(self): return 3 if self.mod.mtype == MODEL_3D else 2
    @property
    def size(self): return self.mod.size
    @size.setter
    def size(self, long val):
        self.mod.size = val
        self.mod.center = val // 2
        self.mod.vol = val**self.ndim
    @property
    def center(self): return self.mod.center
    @property
    def vol(self): return self.mod.vol
    @property
    def num_modes(self): return self.mod.num_modes
    @num_modes.setter
    def num_modes(self, int val): self.mod.num_modes = val

    @property
    def model1(self): return np.asarray(<double[:self.num_modes*self.size**self.ndim]>self.mod.model1).reshape((self.num_modes,) + self.ndim*(self.size,))
    @property
    def model2(self): return np.asarray(<double[:self.num_modes*self.size**self.ndim]>self.mod.model2).reshape((self.num_modes,) + self.ndim*(self.size,))
    @property
    def inter_weight(self): return np.asarray(<double[:self.num_modes*self.size**self.ndim]>self.mod.inter_weight).reshape((self.num_modes,) + self.ndim*(self.size,))
