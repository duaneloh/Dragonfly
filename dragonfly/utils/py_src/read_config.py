'''Module containing various functions used to parse configuration files'''

from __future__ import print_function
import logging
import os
from collections import OrderedDict
from six.moves import configparser
import numpy as np

class MultiOrderedDict(OrderedDict):
    def __init__(self):
        super(MultiOrderedDict, self).__init__()

    def __setitem__(self, key, value):
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super(MultiOrderedDict, self).__setitem__(key, value)

def get_param(config_file, section, tag):
    '''Get config file parameter
    Use get_filename() for getting file names rather than this general function
    '''
    config = configparser.ConfigParser()
    config.read(config_file)
    return config.get(section, tag)

def get_multi_params(config_file, section, tag):
    '''Gets parameter defined multiple times
    Returns list of all values if more than one
    '''
    config = configparser.RawConfigParser(dict_type=MultiOrderedDict)
    config.read(config_file)
    return config.get(section, tag)

def get_filename(config_file, section, tag):
    '''Get filename from config file
    Resolves ::: symlinks
    '''
    param = get_param(config_file, section, tag)
    if ":::" in param:
        [subsec, subtag] = param.split(":::")
        param = get_filename(config_file, subsec, subtag)
    return param

def get_detector_config(config_file, show=False):
    '''Get detector parameters from config file
    Generates and returns a params dictionary
    '''
    config = configparser.ConfigParser()
    config.read(config_file)
    params = OrderedDict()
    params['wavelength'] = config.getfloat('parameters', 'lambda')
    params['detd'] = config.getfloat('parameters', 'detd')
    detstr = config.get('parameters', 'detsize').split(' ')
    if len(detstr) == 1:
        params['dets_x'] = int(detstr[0])
        params['dets_y'] = int(detstr[0])
        params['detsize'] = int(detstr[0])
    else:
        params['dets_x'] = int(detstr[0])
        params['dets_y'] = int(detstr[1])
        params['detsize'] = max(params['dets_x'], params['dets_y'])
    params['pixsize'] = config.getfloat('parameters', 'pixsize')
    params['stoprad'] = config.getfloat('parameters', 'stoprad')
    params['polarization'] = config.get('parameters', 'polarization')

    # Optional arguments
    try:
        params['ewald_rad'] = config.getfloat('parameters', 'ewald_rad')
    except configparser.NoOptionError:
        params['ewald_rad'] = params['detd'] / params['pixsize']

    try:
        params['mask_fname'] = config.get('make_detector', 'in_mask_file')
    except (configparser.NoOptionError, configparser.NoSectionError):
        params['mask_fname'] = None

    try:
        detcstr = config.get('make_detector', 'center').split(' ')
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

def compute_polarization(polarization, polx, poly, norm):
    '''Returns polarization given pixel coordinates and type
    
    Parameters:
        polarization: Can be 'x', 'y' or 'none'
        polx, poly: x and y coordinates of pixel
        norm: Distance of pixel from interaction point
    '''
    if polarization.lower() == 'x':
        return 1. - (polx**2)/(norm**2)
    elif polarization.lower() == 'y':
        return 1. - (poly**2)/(norm**2)
    elif polarization.lower() == 'none':
        return 1. - (polx**2 + poly**2)/(2*norm**2)
    logging.info('Please set the polarization direction as x, y or none!')
    return None

def read_gui_config(gui, section):
    ''' Read config file parameters needed for GUI operation
    '''
    # Defaults
    gui.polar_params = ['5', '60', '2.', '10.']
    gui.class_fname = 'my_classes.dat'
    gui.stack_size = 0

    # Photons file list
    try:
        pfile = get_filename(gui.config_file, section, 'in_photons_file')
        print('Using in_photons_file: %s' % pfile)
        gui.photons_list = [pfile]
    except configparser.NoOptionError:
        plist = get_filename(gui.config_file, section, 'in_photons_list')
        print('Using in_photons_list: %s' % plist)
        with open(plist, 'r') as fptr:
            gui.photons_list = [line.rstrip() for line in fptr.readlines()]
            gui.photons_list = [line for line in gui.photons_list if line]
    gui.num_files = len(gui.photons_list)

    # Detector file list
    try:
        dfile = get_filename(gui.config_file, section, 'in_detector_file')
        print('Using in_detector_file: %s' % dfile)
        gui.det_list = [dfile]
    except configparser.NoOptionError:
        dlist = get_filename(gui.config_file, section, 'in_detector_list')
        print('Using in_detector_list: %s' % dlist)
        with open(dlist, 'r') as fptr:
            gui.det_list = [line.rstrip() for line in fptr.readlines()]
            gui.det_list = [line for line in gui.det_list if line]
    if len(gui.det_list) > 1 and len(gui.det_list) != len(gui.photons_list):
        raise ValueError('Different number of detector and photon files')

    # Only used with old detector file
    try:
        prm = get_detector_config(gui.config_file)
        gui.ewald_rad = prm['ewald_rad']
        gui.detd = prm['detd']/prm['pixsize']
    except (configparser.NoOptionError, configparser.NoSectionError):
        gui.ewald_rad = None
        gui.detd = None

    # Output folder
    try:
        output_folder = get_filename(gui.config_file, section, 'output_folder')
    except configparser.NoOptionError:
        output_folder = 'data/'
    gui.output_folder = os.path.realpath(output_folder)

    # For specific sections
    if section == 'emc':
        # Frame blacklist
        try:
            gui.blacklist = np.loadtxt(get_filename(gui.config_file, 'emc', 'blacklist_file'),
                                       dtype='u1')
        except configparser.NoOptionError:
            gui.blacklist = None

        # Log file
        gui.log_fname = get_filename(gui.config_file, 'emc', 'log_file')

        # Need scaling
        try:
            gui.need_scaling = bool(int(get_param(gui.config_file, 'emc', 'need_scaling')))
        except configparser.NoOptionError:
            gui.need_scaling = False
    elif section == 'classifier':
        # Polar conversion parameters
        try:
            gui.polar_params = get_param(gui.config_file, 'classifier', 'polar_params').split()
        except configparser.NoOptionError:
            gui.polar_params = ['5', '60', '2.', '10.']

        # Class list file
        try:
            gui.class_fname = get_filename(gui.config_file, 'classifier', 'in_class_file')
        except configparser.NoOptionError:
            gui.class_fname = 'my_classes.dat'

        # Check whether slices
        try:
            gui.stack_size = int(get_param(gui.config_file, 'classifier', 'stack_size'))
        except configparser.NoOptionError:
            gui.stack_size = 0
