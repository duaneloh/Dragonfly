'''Module to generate and manipulate uniformly sampled 3D quaternions'''

from __future__ import print_function
import sys
import os
from socket import gethostname
import itertools
import numpy as np
import pandas
from scipy.spatial import distance

NUM_VERT = 120
NUM_EDGE = 720
NUM_FACE = 1200
NUM_CELL = 600
NNN = 12

cimport numpy as np
from libc.stdlib cimport malloc, calloc, free
from . cimport quaternion as c_quat
from .quaternion cimport Quaternion

cdef class Quaternion:
    def __init__(self, int num_div=0, point_group=''):
        self.quat = <c_quat.quaternion*> malloc(sizeof(c_quat.quaternion))
        self.quat.num_div = num_div
        self.quat.num_rot = 0
        self.quat.octahedral_flag = (point_group == 'S4')
        self.quat.icosahedral_flag = (point_group == 'A5')
        self.quat.num_rot_p = 0
        self.quat.quats = NULL
        self.reduced = False

        if num_div > 0:
            self.generate(num_div)

    def generate(self, int num_div):
        self.reduced = False
        return c_quat.quat_gen(num_div, self.quat)

    def save(self, fname):
        if self.quat.quats == NULL:
            raise AttributeError('Generate quaternion first before saving')

        with open(fname, 'w') as fptr:
            fptr.write('%d\n' % self.quat.num_rot)
            np.savetxt(fptr, self.quats, fmt='%+17.15f')

    def parse(self, fname):
        cdef int i

        df = pandas.read_csv(fname, sep='\s+', skiprows=1, header=None, dtype='f8')
        quats = df.values
        self.quat.num_rot = quats.shape[0]

        if quats.shape[1] == 5:
            quats[:,4] /= quats[:,4].sum()
        elif quats.shape[1] == 4:
            quats = np.pad(quats, ((0,0),(0,1)), mode='constant',
                           constant_values=1./self.num_rot)
        else:
            raise ValueError('Unknown shape for data in %s'%fname)
        self.quat.quats = <double*> malloc(self.num_rot * 5 * sizeof(double))
        quats = quats.ravel()
        for i in np.arange(5*self.num_rot):
            self.quat.quats[i] = quats[i]

    def divide(self, rank, num_proc, num_modes=1, num_nonrot_modes=0):
        tot_num_rot = num_modes * self.num_rot + num_nonrot_modes
        self.num_rot_p = tot_num_rot / num_proc
        if rank < (tot_num_rot % num_proc):
            self.quat.num_rot_p += 1
        if num_proc > 1:
            sys.stderr.write("%d: %s: num_rot_p = %d/%d\n" % (rank, gethostname(), self.num_rot_p, tot_num_rot))
            sys.stderr.flush()
        return self.quat.num_rot_p

    def reduce_icosahedral(self, return_sym=False):
        if self.quat.quats == NULL:
            raise AttributeError('Generate quaternion first before reducing')
        if self.reduced:
            print('Already reduced')
            return self.num_rot

        cdef int r, t

        quats = self.quats
        # Icosahedral point group quaternions are just the vertex ones
        # Keep quaternions whose closest vertex_quat is the identity
        vert_dist = distance.cdist(quats[:60,:4], quats[60:,:4])
        sel = np.where(vert_dist.argmin(axis=0) == 8)[0]
        self.num_rot = 1 + sel.shape[0]

        for t in range(5):
            self.quat.quats[t] = self.quat.quats[8*5 + t]
        for r in range(1, self.num_rot):
            for t in range(5):
                self.quat.quats[r*5 + t] = self.quat.quats[(sel[r-1]+60)*5 + t]

        self.reduced = True
        if return_sym:
            return quats[:60,:4]
        else:
            return self.num_rot

    def reduce_octahedral(self, return_sym=False):
        if self.quat.quats == NULL:
            raise AttributeError('Generate quaternion first before reducing')
        if self.reduced:
            print('Already reduced')
            return self.num_rot

        # Generate cubic point group quaternions
        cube_quat = np.zeros((24, 4))

        # -- neut and inv2 (1x + 3x)
        cube_quat[:4] = np.identity(4)

        # -- rot3 (8x)
        signs3 = np.stack(np.meshgrid([-1., 1.], [-1., 1.], [-1., 1.], indexing='ij'), -1).reshape(-1, 3)
        cube_quat[4:12] = 0.5
        cube_quat[4:12,1:] *= signs3

        # -- rot1 (6x)
        val = np.sqrt(0.5)
        cube_quat[12:18, 0] = val
        cube_quat[12:15, 1:] = np.identity(3) * val
        cube_quat[15:18, 1:] = np.identity(3) * val * -1

        # -- rot2 (6x)
        for i, pos in enumerate(itertools.permutations(range(3), 2)):
            cube_quat[18 + i, np.array(pos) + 1] = [val, -val] if pos[0] < pos[1] else [val, val]

        cdef int r, t
        quats = self.quats

        # Keep quaternions whose closest cube_quat is the identity
        cube_dist = distance.cdist(cube_quat, quats[:,:4])
        sel = np.where(cube_dist.argmin(axis=0) == 0)[0]
        self.num_rot = sel.shape[0]
        for r in range(self.num_rot):
            for t in range(5):
                self.quat.quats[r*5 + t] = self.quat.quats[sel[r]*5 + t]

        self.reduced = True
        if return_sym:
            return cube_quat
        else:
            return self.num_rot

    def voronoi_subset(self, int coarse_num_div):
        if self.num_rot == 0:
            print('Generate fine quaternions first')
            return
        cdef c_quat.quaternion* coarse
        coarse = <c_quat.quaternion*> calloc(1, sizeof(c_quat.quaternion))
        coarse.num_div = coarse_num_div
        c_quat.quat_gen(coarse_num_div, coarse)

        nearest = np.empty(self.num_rot, dtype='i4')
        cdef int[:] nearest_view = nearest
        c_quat.voronoi_subset(coarse, self.quat, &nearest_view[0])

        free(coarse.quats)
        free(coarse)

        return nearest

    def free(self):
        if self.quat == NULL:
            return
        if self.quat.quats != NULL: free(self.quat.quats)
        free(self.quat)
        self.quat = NULL

    @property
    def num_div(self): return self.quat.num_div
    @property
    def num_rot(self): return self.quat.num_rot
    @num_rot.setter
    def num_rot(self, val): self.quat.num_rot = val
    @property
    def num_rot_p(self): return self.quat.num_rot_p
    @num_rot_p.setter
    def num_rot_p(self, val): self.quat.num_rot_p = val
    @property
    def reduced(self): return bool(self.quat.reduced)
    @reduced.setter
    def reduced(self, val): self.quat.reduced = int(val)
    @property
    def icosahedral_flag(self): return bool(self.quat.icosahedral_flag)
    @property
    def octahedral_flag(self): return bool(self.quat.octahedral_flag)
    @property
    def quats(self): return np.asarray(<double[:self.num_rot*5]> self.quat.quats).reshape(-1,5) if self.quat.quats != NULL else None
    @quats.setter
    def quats(self, arr):
        if len(arr.shape) != 2 or arr.shape[1] != 5:
            raise ValueError('quats must be  2D array of shape (N, 5)')
        if arr.dtype != 'f8':
            raise TypeError('quats must be double precision floats (float64)')

        self.quat.quats = <double*> malloc(arr.size * sizeof(double))
        for i in range(arr.size):
            self.quat.quats[i] = arr.ravel()[i]
        self.quat.num_rot = arr.shape[0]

