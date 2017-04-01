import sys
import numpy as np

class Slice_generator():
    def __init__(self, geom, quat_fname, folder='data/'):
        self.det = np.array([geom.qx, geom.qy, geom.qz]).T
        self.mask = geom.mask
        self.frame_shape = geom.frame_shape
        self.ix = geom.x
        self.iy = geom.y
        self.quat = np.loadtxt(quat_fname, skiprows=1, usecols=(0,1,2,3))
        self.folder = folder
        self.current_iteration = -1

    def init_model(self, iteration):
        self.model = np.fromfile('%s/output/intens_%.3d.bin' % (self.folder,iteration), '=f8')
        size = int(np.round(self.model.shape[0]**(1./3.)))
        self.model = self.model.reshape(size,size,size)
        self.size = size
        self.rmax = np.fromfile('%s/orientations/orientations_%.3d.bin' % (self.folder, iteration), '=i4')
        try:
            self.scale = np.loadtxt('%s/scale/scale_%.3d.dat' % (self.folder, iteration))
        except IOError:
            self.scale = np.ones(self.rmax.shape)
        self.current_iteration = iteration

    def gen_rot_matrix(self, num):
        quat = self.quat[num]
        qx = quat[1]; qy = quat[2]; qz = quat[3]; qw = quat[0]
        rot = np.zeros((3,3))
        rot[0,0] = 1-2*qy*qy-2*qz*qz ; rot[0,1] = 2*qx*qy - 2*qz*qw ; rot[0,2] = 2*qx*qz + 2*qy*qw
        rot[1,0] = 2*qx*qy + 2*qz*qw ; rot[1,1] = 1-2*qx*qx-2*qz*qz ; rot[1,2] = 2*qy*qz - 2*qx*qw
        rot[2,0] = 2*qx*qz - 2*qy*qw ; rot[2,1] = 2*qy*qz + 2*qx*qw ; rot[2,2] = 1-2*qx*qx-2*qy*qy
        return rot

    def get_slice(self, iteration, num):
        if iteration != self.current_iteration:
            self.init_model(iteration)
        rot_matrix = self.gen_rot_matrix(self.rmax[num])
        rot_coords = np.dot(rot_matrix, self.det.T)
        rot_coords = np.round((rot_coords + self.size/2)).astype('i4')
        rot_coords[(rot_coords<0) | (rot_coords>self.size-1)] = 0
        sl = self.model[rot_coords[0], rot_coords[1], rot_coords[2]]
        im = np.zeros(self.frame_shape)
        np.add.at(im, [self.ix, self.iy], sl)
        return im * self.mask * self.scale[num]

