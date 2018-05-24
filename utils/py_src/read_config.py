import numpy as np
import logging
import ConfigParser
import os
from collections import OrderedDict

class MultiOrderedDict(OrderedDict):
    def __setitem__(self, key, value):
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super(OrderedDict, self).__setitem__(key, value)

def get_param(config_file, section, tag):
    config      = ConfigParser.ConfigParser()
    config.read(config_file)
    return config.get(section, tag)

def get_multi_params(config_file, section, tag):
    config      = ConfigParser.RawConfigParser(dict_type=MultiOrderedDict)
    config.read(config_file)
    return config.get(section, tag)

def get_filename(config_file, section, tag):
    param   = get_param(config_file, section, tag)
    if ":::" in param:
        [s, t] = param.split(":::")
        param = get_filename(config_file, s, t)
    return param

def get_detector_config(config_file, show=False):
    config      = ConfigParser.ConfigParser()
    config.read(config_file)
    params      = OrderedDict()
    params['wavelength']   = config.getfloat('parameters', 'lambda')
    params['detd']         = config.getfloat('parameters', 'detd')
    detstr = config.get('parameters', 'detsize').split(' ')
    if len(detstr) == 1:
        params['dets_x']   = int(detstr[0])
        params['dets_y']   = int(detstr[0])
        params['detsize']  = int(detstr[0])
    else:
        params['dets_x']   = int(detstr[0])
        params['dets_y']   = int(detstr[1])
        params['detsize']  = max(params['dets_x'], params['dets_y'])
    params['pixsize']      = config.getfloat('parameters', 'pixsize')
    params['stoprad']      = config.getfloat('parameters', 'stoprad')
    params['polarization'] = config.get('parameters', 'polarization')

    # Optional arguments
    try:
        params['ewald_rad'] = config.getfloat('parameters', 'ewald_rad')
    except ConfigParser.NoOptionError:
        params['ewald_rad'] = params['detd'] / params['pixsize']

    try:
        params['mask_fname'] = config.get('make_detector', 'in_mask_file')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        params['mask_fname'] = None

    try:
        detcstr = config.get('make_detector', 'center').split(' ')
        if len(detstr) == 1:
            params['detc_x'] = int(detcstr[0])
            params['detc_y'] = int(detcstr[0])
        else:
            params['detc_x'] = int(detcstr[0])
            params['detc_y'] = int(detcstr[1])
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        params['detc_x']   = (params['dets_x']-1)/2.
        params['detc_y']   = (params['dets_y']-1)/2.
    
    if show:
        for k,v in params.items():
            #print '{:<15}:{:10.4f}'.format(k, v)
            #print '{:<15}:{:>10}'.format(k, v)
            logging.info('{:<15}:{:>10}'.format(k, v))
    return params

def compute_q_params(det_dist, dets_x, dets_y, pix_size, in_wavelength, ewald_rad, show=False):
    """
    Resolution computed in inverse Angstroms, crystallographer's convention
    In millimeters: det_dist, pix_size
    In Angstroms:   in_wavelength
    In pixels:      dets_x, dets_y

    """
    params      = OrderedDict()
    half_x      = pix_size * int((dets_x-1)/2)
    half_y      = pix_size * int((dets_y-1)/2)
    max_angle   = np.arctan(np.sqrt(half_x**2 + half_y**2) / det_dist)
    min_angle   = np.arctan(pix_size / det_dist)
    q_max       = 2. * np.sin(0.5 * max_angle) / in_wavelength
    q_sep       = 2. * np.sin(0.5 * min_angle) / in_wavelength * (det_dist / ewald_rad / pix_size)
    fov_in_A    = 1. / q_sep
    half_p_res  = 0.5 / q_max
    params['max_angle'] = max_angle
    params['min_angle'] = min_angle
    params['q_max']     = q_max
    params['q_sep']     = q_sep
    params['fov_in_A']  = fov_in_A
    params['half_p_res']= half_p_res

    if show:
        for k,v in params.items():
            #print '{:<15}:{:10.4f}'.format(k, v)
            logging.info('{:<15}:{:10.4f}'.format(k, v))
        #print '{:<15}:{:10.4f}'.format("voxel-length of reciprocal volume", fov_in_A/half_p_res)
        logging.info('{:<15}:{:10.4f}'.format("voxel-length of reciprocal volume", fov_in_A/half_p_res))
    return params

def compute_polarization(polarization, px, py, norm):
    if polarization.lower() == 'x':
        return 1. - (px*px)/(norm*norm)
    elif polarization.lower() == 'y':
        return 1. - (py*py)/(norm*norm)
    elif polarization.lower() == 'none':
        return 1. - (px*px + py*py)/(2*norm*norm)
    else:
        #print 'Please set the polarization direction as x, y or none!'
        logging.info('Please set the polarization direction as x, y or none!')

def read_gui_config(gui, section):
    ''' Read config file parameters needed for GUI operation
    '''
    # Photons file list
    try:
        pfile = get_filename(gui.config_file, section, 'in_photons_file')
        print 'Using in_photons_file: %s' % pfile
        gui.photons_list = [pfile]
    except ConfigParser.NoOptionError:
        plist = get_filename(gui.config_file, section, 'in_photons_list')
        print 'Using in_photons_list: %s' % plist
        with open(plist, 'r') as f:
            gui.photons_list = map(lambda x: x.rstrip(), f.readlines())
            gui.photons_list = [line for line in gui.photons_list if line]
    gui.num_files = len(gui.photons_list)
    
    # Detector file list
    try:
        dfile = get_filename(gui.config_file, section, 'in_detector_file')
        print 'Using in_detector_file: %s' % dfile
        gui.det_list = [dfile]
    except ConfigParser.NoOptionError:
        dlist = get_filename(gui.config_file, section, 'in_detector_list')
        print 'Using in_detector_list: %s' % dlist
        with open(dlist, 'r') as f:
            gui.det_list = map(lambda x: x.rstrip(), f.readlines())
            gui.det_list = [line for line in gui.det_list if line]
    if len(gui.det_list) > 1 and len(gui.det_list) != len(gui.photons_list):
        raise ValueError('Different number of detector and photon files')
    
    # Output folder
    try:
        output_folder = get_filename(gui.config_file, 'emc', 'output_folder')
    except ConfigParser.NoOptionError:
        output_folder = 'data/'
    gui.output_folder = os.path.realpath(output_folder)
    
    # Frame blacklist
    try:
        gui.blacklist = np.loadtxt(get_filename(gui.config_file, 'emc', 'blacklist_file'), dtype='u1')
    except ConfigParser.NoOptionError:
        gui.blacklist = None
    
    # Only used with old detector file
    try:
        pm = get_detector_config(gui.config_file)
        gui.ewald_rad = pm['ewald_rad']
        gui.detd = pm['detd']/pm['pixsize']
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        gui.ewald_rad = None
        gui.detd = None
    
    # Log file
    gui.log_fname = get_filename(gui.config_file, 'emc', 'log_file')
    
