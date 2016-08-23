import numpy as np
import logging
import ConfigParser
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

def compute_q_params(det_dist, dets_x, dets_y, pix_size, in_wavelength, show=False):
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
    q_sep       = 2. * np.sin(0.5 * min_angle) / in_wavelength
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
