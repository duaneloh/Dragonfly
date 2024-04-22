'''Module containing class to generate tomographic slices'''

import os

import h5py
import dragonfly

from .read_config import MyConfigParser

class SliceGenerator(object):
    '''Class to generate slices from 3D intensity distribution for given orientation

    Requires config file from an EMC reconstruction

    Methods:
        get_slice(iteration, frame_number) - Return predicted detector intensity for iteration'th\
            reconstruction for given frame

    NOTE: This produces the tomogram for the most likely orientation only.
    '''
    def __init__(self, config_fname):
        config = MyConfigParser()
        config.read(config_fname)

        self.det = dragonfly.Detector(config.get_filename('emc', 'in_detector_file'))
        self.quat = dragonfly.Quaternion()
        self.quat.from_config(config_fname, 'emc')
        self.recon_type = config.get('emc', 'recon_type', fallback='3d').lower()
        self.folder = config.get_filename('emc', 'output_folder', fallback='data/')
        self.need_scaling = config.getboolean('emc', 'need_scaling', fallback=False)

        self.current_iteration = -1
        self.model = self.stats = None

    def _init_model(self, iteration):
        self.stats = {'rmax': None, 'scale': None, 'info': None}

        h5model_fname = '%s/output_%.3d.h5' % (self.folder, iteration)
        print('Parsing comparison output:', h5model_fname)
        with h5py.File(h5model_fname, 'r') as fptr:
            size = fptr['intens'].shape[-1]
            self.num_modes = fptr['intens'].shape[0]
            self.stats['rmax'] = fptr['orientations'][:]
            self.stats['info'] = fptr['mutual_info'][:]
            if self.need_scaling:
                self.stats['scale'] = fptr['scale'][:]
        self.model = dragonfly.Model(size, self.num_modes, model_type=self.recon_type)
        self.model.allocate(h5model_fname)

        self.current_iteration = iteration

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
        mode_num = self.stats['rmax'][num] % self.num_modes
        quat_num = self.stats['rmax'][num] // self.num_modes
        dslice = self.model.slice_gen(self.quat.quats[quat_num], self.det, mode=mode_num)

        if raw:
            return dslice, self.stats['info'][num]
        return self.det.assemble_frame(dslice), self.stats['info'][num]

    def get_quat(self, iteration, num):
        '''Return best match quaternion for given iteration and frame number'''
        if iteration != self.current_iteration:
            self._init_model(iteration)
        return tuple(self.quat.quats[self.stats['rmax'][num] // self.num_modes])
