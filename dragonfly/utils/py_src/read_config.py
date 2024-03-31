'''Module containing various functions used to parse configuration files'''

import logging
import os.path as op
from collections import OrderedDict
import configparser
import numpy as np

class MyConfigParser(configparser.ConfigParser):
    def read(self, filename):
        self.config_folder = op.dirname(filename)
        super().read(filename)

    def get_filename(self, section, option, *, fallback=''):
        fname = self.get(section, option, fallback=fallback)
        if fname == '':
            raise configparser.NoOptionError(section, option)
        if fname is None:
            return fname

        if ":::" in fname:
            [subsec, subtag] = fname.split(":::")
            fname = self.get_filename(subsec, subtag)
        return op.join(self.config_folder, fname)

    def get_detector_config(self, show=False):
        '''Get detector parameters from config file
        Generates and returns a params dictionary
        '''
        params = OrderedDict()
        params['wavelength'] = self.getfloat('parameters', 'lambda')
        params['detd'] = self.getfloat('parameters', 'detd')
        detstr = self.get('parameters', 'detsize').split(' ')
        if len(detstr) == 1:
            params['dets_x'] = int(detstr[0])
            params['dets_y'] = int(detstr[0])
            params['detsize'] = int(detstr[0])
        else:
            params['dets_x'] = int(detstr[0])
            params['dets_y'] = int(detstr[1])
            params['detsize'] = max(params['dets_x'], params['dets_y'])
        params['pixsize'] = self.getfloat('parameters', 'pixsize')
        params['stoprad'] = self.getfloat('parameters', 'stoprad')
        params['polarization'] = self.get('parameters', 'polarization')

        # Optional arguments
        try:
            params['ewald_rad'] = self.getfloat('parameters', 'ewald_rad')
        except configparser.NoOptionError:
            params['ewald_rad'] = params['detd'] / params['pixsize']

        try:
            params['mask_fname'] = self.get('make_detector', 'in_mask_file')
        except (configparser.NoOptionError, configparser.NoSectionError):
            params['mask_fname'] = None

        try:
            detcstr = self.get('make_detector', 'center').split(' ')
            if len(detstr) == 1:
                params['detc_x'] = int(detcstr[0])
                params['detc_y'] = int(detcstr[0])
            else:
                params['detc_x'] = int(detcstr[0])
                params['detc_y'] = int(detcstr[1])
        except (configparser.NoOptionError, configparser.NoSectionError):
            params['detc_x'] = (params['dets_x']-1)/2.
            params['detc_y'] = (params['dets_y']-1)/2.

        if show:
            for key, val in params.items():
                #logging.info('{:<15}:{:>10}'.format(key, val))
                logging.info('%15s:%-10s', key, str(val))
        return params

def compute_q_params(det_dist, dets_x, dets_y, pix_size, in_wavelength, ewald_rad, show=False):
    """
    Resolution computed in inverse Angstroms, crystallographer's convention
    In millimeters: det_dist, pix_size
    In Angstroms:   in_wavelength
    In pixels:      dets_x, dets_y
    """
    params = OrderedDict()
    half_x = pix_size * int((dets_x-1)/2)
    half_y = pix_size * int((dets_y-1)/2)
    params['max_angle'] = np.arctan(np.sqrt(half_x**2 + half_y**2) / det_dist)
    params['min_angle'] = np.arctan(pix_size / det_dist)
    params['q_max'] = 2. * np.sin(0.5 * params['max_angle']) / in_wavelength
    params['q_sep'] = 2. * np.sin(0.5 * params['min_angle']) / in_wavelength\
                      * (det_dist / ewald_rad / pix_size)
    params['fov_in_A'] = 1. / params['q_sep']
    params['half_p_res'] = 0.5 / params['q_max']

    if show:
        for key, val in params.items():
            logging.info('%15s:%10.4f', key, val)
        logging.info('%15s:%10.4f',
                     'voxel-length or reciprocal volume',
                     params['fov_in_A']/params['half_p_res'])
    return params

def read_gui_config(gui, section):
    ''' Read config file parameters needed for GUI operation
    '''
    config = MyConfigParser()
    config.read(gui.config_file)
    
    # Photons file list
    try:
        pfile = config.get_filename(section, 'in_photons_file')
        print('Using in_photons_file: %s' % pfile)
        gui.photons_list = [pfile]
    except configparser.NoOptionError:
        plist = config.get_filename(section, 'in_photons_list')
        print('Using in_photons_list: %s' % plist)
        with open(plist, 'r') as fptr:
            gui.photons_list = [line.rstrip() for line in fptr.readlines()]
            gui.photons_list = [line for line in gui.photons_list if line]
    gui.num_files = len(gui.photons_list)

    # Detector file list
    try:
        dfile = config.get_filename(section, 'in_detector_file')
        print('Using in_detector_file: %s' % dfile)
        gui.det_list = [dfile]
    except configparser.NoOptionError:
        dlist = config.get_filename(section, 'in_detector_list')
        print('Using in_detector_list: %s' % dlist)
        with open(dlist, 'r') as fptr:
            gui.det_list = [line.rstrip() for line in fptr.readlines()]
            gui.det_list = [line for line in gui.det_list if line]
    if len(gui.det_list) > 1 and len(gui.det_list) != len(gui.photons_list):
        raise ValueError('Different number of detector and photon files')

    # Output folder
    output_folder = config.get_filename(section, 'output_folder', fallback='data/')
    gui.output_folder = op.realpath(output_folder)

    # For specific sections
    if section == 'emc':
        # Frame blacklist
        b_fname = config.get_filename('emc', 'blacklist_file', fallback=None)
        if b_fname is None:
            gui.blacklist = None
        else:
            gui.blacklist = np.loadtxt(b_fname, dtype='u1')

        gui.log_fname = config.get_filename('emc', 'log_file', fallback='logs/EMC.log')
        gui.need_scaling = config.getboolean('emc', 'need_scaling')
    elif section == 'classifier':
        gui.polar_params = config.get('classifier', 'polar_params', fallback=['5', '60', '2.', '10.'])
        gui.class_fname = config.get_filename('classifier', 'in_class_file', fallback='my_classes.dat')
        gui.stack_size = config.getint('classifier', 'stack_size', fallback=0)
