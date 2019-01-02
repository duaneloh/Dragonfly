'''Module containing class to generate tomographic slices'''

from __future__ import print_function
import numpy as np

class SliceGenerator(object):
    '''Class to generate slices from 3D intensity distribution for given orientation
    Requires:
        Detector object - Instance of DetReader
        Quaternion filename - Typically made using utils/make_quaternion
        Output folder [Optional] - Path to data folder where reconstruction results are stored
    Methods:
        get_slice(iteration, frame_number) - Return predicted detector intensity for iteration'th
            reconstruction for given frame

    NOTE: This produces the tomogram for the most likely orientation only.
    '''
    def __init__(self, geom, quat_fname, folder='data/', need_scaling=False):
        self.geom = geom
        self.quat = np.loadtxt(quat_fname, skiprows=1, usecols=(0, 1, 2, 3))
        self.folder = folder
        self.current_iteration = -1
        self.need_scaling = need_scaling

        self.model = self.stats = None

    def _init_model(self, iteration):
        model_fname = '%s/output/intens_%.3d.bin' % (self.folder, iteration)
        print('Parsing comparison model:', model_fname)
        self.model = np.fromfile(model_fname, '=f8')
        size = int(np.round(self.model.shape[0]**(1./3.)))
        self.model = self.model.reshape(3 * (size,))

        self.stats = {'rmax': None, 'scale': None, 'info': None}
        self.stats['rmax'] = np.fromfile('%s/orientations/orientations_%.3d.bin' %
                                         (self.folder, iteration), '=i4')
        if self.need_scaling:
            try:
                self.stats['scale'] = np.loadtxt('%s/scale/scale_%.3d.dat' %
                                                 (self.folder, iteration))
            except IOError:
                self.stats['scale'] = np.ones(self.stats['rmax'].shape)
        else:
            self.stats['scale'] = np.ones(self.stats['rmax'].shape)
        self.stats['info'] = np.loadtxt('%s/mutualInfo/info_%.3d.dat' % (self.folder, iteration))

        self.current_iteration = iteration

    def _gen_rot_matrix(self, num):
        qw, qx, qy, qz = tuple(self.quat[num]) # pylint: disable=C0103
        rot = np.zeros((3, 3))
        rot[0] = [1-2*qy*qy-2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw]
        rot[1] = [2*qx*qy + 2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz - 2*qx*qw]
        rot[2] = [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1-2*qx*qx-2*qy*qy]
        return np.linalg.inv(rot)

    def get_slice(self, iteration, num, raw=False):
        '''Get tomographic slice for given iteration and frame number
        Parameters:
            iteration - Reconstruction iteration number
            num - Frame number
            raw [optional] - Whether to return unassembled slice
        Returns:
            Assembled detector slice, mutual information for that iteration
        '''
        if iteration != self.current_iteration:
            self._init_model(iteration)
        rot_matrix = self._gen_rot_matrix(self.stats['rmax'][num])

        size = self.model.shape[0]
        coords = np.array([self.geom.qx, self.geom.qy, self.geom.qz])
        rot_coords = np.dot(rot_matrix, coords)
        rot_coords = np.round((rot_coords + size/2)).astype('i4')
        rot_coords[(rot_coords < 0) | (rot_coords > size-1)] = 0
        dslice = self.model[rot_coords[0], rot_coords[1], rot_coords[2]]

        if raw:
            return dslice, self.stats['info'][num]
        imslice = np.zeros(self.geom.frame_shape)
        np.add.at(imslice, [self.geom.x, self.geom.y], dslice)
        return imslice * self.geom.mask * self.stats['scale'][num], self.stats['info'][num]

    def get_quat(self, iteration, num):
        '''Return best match quaternion for given iteration and frame number'''
        if iteration != self.current_iteration:
            self._init_model(iteration)
        return tuple(self.quat[self.stats['rmax'][num]])
