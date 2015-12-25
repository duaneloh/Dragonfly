import numpy as np
import ConfigParser
import sys
from collections import OrderedDict

################################################################################
# Functions useful for later bits
################################################################################

def compute_reciprocal_params(det_dist, det_size, pix_size, in_wavelength):
    """
    Resolution computed in inverse Angstroms, crystallographer's convention
    In millimeters: in_det_dis, in_det_size, in_pix_size
    In Angstroms:   photon wavelength

    """
    det_max_half_len = pix_size * int((det_size-1)/2.)
    params      = OrderedDict()
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
    return params

################################################################################
# Script begins
################################################################################

if __name__ == "__main__":
    # Prints a report of exp_config.dat
    config = ConfigParser.ConfigParser()
    config.read(sys.argv[1])

    pdbFile     = config.get('files', 'pdb')
    wavelength  = config.getfloat('parameters', 'lambda')
    detd        = config.getfloat('parameters', 'detd')
    detsize     = config.getint('parameters', 'detsize')
    pixsize     = config.getfloat('parameters', 'pixsize')
    stoprad     = config.getfloat('parameters', 'stoprad')
    params      = compute_reciprocal_params(detd, detsize, pixsize, wavelength)

    for k,v in params.items():
        print '{:<15}:{:10.4f}'.format(k, v)
