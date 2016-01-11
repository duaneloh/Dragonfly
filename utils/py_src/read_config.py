import numpy as np
import logging
import ConfigParser
from collections import OrderedDict

def get_param(config_file, section, tag):
    config      = ConfigParser.ConfigParser()
    config.read(config_file)
    return config.get(section, tag)

def get_filename(config_file, section, tag):
    param   = get_param(config_file, section, tag)
    if ":::" in param:
        [s, t] = param.split(":::")
        param = get_param(config_file, s, t)
    return param

def get_detector_config(config_file, show=False):
    config      = ConfigParser.ConfigParser()
    config.read(config_file)
    params      = OrderedDict()
    params['wavelength']   = config.getfloat('parameters', 'lambda')
    params['detd']         = config.getfloat('parameters', 'detd')
    params['detsize']      = config.getint('parameters', 'detsize')
    params['pixsize']      = config.getfloat('parameters', 'pixsize')
    params['stoprad']      = config.getfloat('parameters', 'stoprad')
    params['polarization'] = config.get('parameters', 'polarization')
    if show:
        for k,v in params.items():
            #print '{:<15}:{:10.4f}'.format(k, v)
            #print '{:<15}:{:>10}'.format(k, v)
            logging.info('{:<15}:{:>10}'.format(k, v))
    return params

def compute_q_params(det_dist, det_size, pix_size, in_wavelength, show=False, squareDetector=True):
    """
    Resolution computed in inverse Angstroms, crystallographer's convention
    In millimeters: det_dist, pix_size
    In Angstroms:   in_wavelength
    Unitless: det_size

    """
    det_max_half_len = pix_size * int((det_size-1)/2.)
    params      = OrderedDict()
    if squareDetector:
        max_angle   = np.arctan(np.sqrt(2.) * det_max_half_len / det_dist)
    else:
        max_angle   = np.arctan(det_max_half_len / det_dist)
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

def compute_polarization(polarization, qx, qy, norm):
    if polarization.lower() == 'x':
        return 1. - (qx*qx)/(norm*norm)
    elif polarization.lower() == 'y':
        return 1. - (qy*qy)/(norm*norm)
    elif polarization.lower() == 'none':
        return 1.
    else:
        #print 'Please set the polarization direction as x, y or none!'
        logging.info('Please set the polarization direction as x, y or none!')
