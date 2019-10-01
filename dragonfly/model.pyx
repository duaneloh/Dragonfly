from __future__ import print_function
import sys
import os
import numpy as np
from scipy import ndimage
import h5py
import pandas

cimport numpy as np

class Model():
    def __init__(self):
        self.size = -1
        self.num_modes = 1
        self.model1 = None
        self.model2 = None
        self.inter_weight = None
        self.scale = None

        self.bgscale = None
        self.rms_change = 0

    def parse_scale(self, fname, bg=False):
        if h5py.is_hdf5(fname):
            with h5py.File(fname, 'r') as f:
                scale = f['scale'][:]
        else:
            scale = pandas.read_csv(fname, header=None).array.ravel()
        
        if bg:
            self.bgscale = scale
        else:
            self.scale = scale

    def normalize_scale(self, frames):
        blist = frames.blacklist
        mean_scale = self.scale[blist==0].mean()
        self.model1 * mean_scale
        self.scale[blist==0] /= mean_scale
        self.rms_change *= mean_scale

    def parse_input(self, fname, model_mean, rank=0, recon_type='3d'):
        ndim = 3 if recon_type == '3d' else 2
        mshape = (self.num_modes,) + ndim*(self.size,)

        self.model1 = np.empty(mshape)
        self.model2 = np.empty(mshape)
        self.inter_weight = np.empty(mshape)

        if os.path.isfile(fname):
            if h5py.is_hdf5(fname):
                with h5py.File(fname, 'r') as f:
                    fshape = f['intens'].shape
                    if fshape == mshape:
                        self.model1[:] = f['intens'][:]
                    elif fshape[1:] != mshape[1:]:
                        raise ValueError('Input model has wrong grid size')
                    elif fshape[0] < mshape[0]:
                        print('More modes in file than expected, only reading in first %d'%self.num_modes)
                        self.model1[:] = f['intens'][:self.num_modes]
                    else:
                        print('Not enough modes in file. Filling rest with white noise')
                        self.model1[:fshape[0]] = f['intens'][:]
                        self.model1[fshape[0]:] = np.random.random((mshape[0]-fshape[0],) + ndim*(self.size,))
            else:
                self.model1[:] = np.fromfile(fname).reshape(mshape)
        else:
            self.model1[:] = np.random.random(mshape) * model_mean

    def slice_gen(self, quat, det, mode=0, rescale=1., recon_type='3d'):
        rot = self._make_rot_quat(quat)
        if recon_type == '3d':
            pix = [det.qx, det.qy, det.qz]
        else:
            pix = [det.cx, det.cy]
        coords = np.dot(rot, pix)
        view = ndimage.map_coordinates(self.model1[mode], coords, 
                                       order=1, prefilter=False,
                                       cval=sys.float_info.min)
        view *= det.corr
        if rescale == 0:
            return view
        else:
            return np.log(view * rescale)

    @staticmethod
    def _make_rot_quat(quaternion, rot=None):
        rot = np.zeros((3, 3))
        q0 = quaternion[0]
        q1 = quaternion[1]
        q2 = quaternion[2]
        q3 = quaternion[3]
        
        q01 = q0*q1
        q02 = q0*q2
        q03 = q0*q3
        q11 = q1*q1
        q12 = q1*q2
        q13 = q1*q3
        q22 = q2*q2
        q23 = q2*q3
        q33 = q3*q3
        
        rot[0][0] = (1. - 2.*(q22 + q33))
        rot[0][1] = 2.*(q12 + q03)
        rot[0][2] = 2.*(q13 - q02)
        rot[1][0] = 2.*(q12 - q03)
        rot[1][1] = (1. - 2.*(q11 + q33))
        rot[1][2] = 2.*(q01 + q23)
        rot[2][0] = 2.*(q02 + q13)
        rot[2][1] = 2.*(q23 - q01)
        rot[2][2] = (1. - 2.*(q11 + q22))

    @staticmethod
    def calculate_size(qmax):
        return 2 * np.ceil(qmax) + 3
