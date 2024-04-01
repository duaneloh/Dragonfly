#!/usr/bin/env python

'''Module to Fourier transform electron densities to generate 3D intensities'''

import sys
import logging
import argparse

import numpy as np

from .py_src import read_config
from .py_src import py_utils
try:
    import pyfftw
    WITH_PYFFTW = True
except ImportError:
    WITH_PYFFTW = False

def make_intens(config_fname, yes=False, verbose=False):
    '''Generate intensity volume from config file parameters'''
    config = read_config.MyConfigParser()
    config.read(config_fname)

    dens_fname = config.get_filename('make_intensities', 'in_density_file')
    intens_fname = config.get_filename('make_intensities', 'out_intensity_file')
    num_threads = config.getint('make_intensities', 'num_threads', fallback=4)

    if yes or py_utils.check_to_overwrite(intens_fname):
        timer = py_utils.MyTimer()
        pm = config.get_detector_config(show=verbose)
        q_pm = read_config.compute_q_params(pm['detd'], pm['dets_x'],
                                            pm['dets_y'], pm['pixsize'],
                                            pm['wavelength'], pm['ewald_rad'], show=verbose)
        timer.reset('Reading experiment parameters', report=verbose)

        fov_len = 2 * int(np.ceil(q_pm['fov_in_A']/q_pm['half_p_res']/2.)) + 3
        logging.info('Volume size: %d', fov_len)
        den = py_utils.read_density(dens_fname)
        min_over = float(fov_len)/den.shape[0]
        if min_over > 12:
            if py_utils.confirm_oversampling(min_over) is False:
                sys.exit(0)
        timer.reset('Reading densities', report=verbose)

        pad_den = np.zeros(3*(fov_len,))
        den_sh = den.shape
        pad_den[:den_sh[0], :den_sh[1], :den_sh[2]] = den.copy()
        if WITH_PYFFTW:
            intens = np.abs(np.fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(
                pad_den, threads=num_threads, planner_effort='FFTW_ESTIMATE')))**2
        else:
            intens = np.abs(np.fft.fftshift(np.fft.fftn(pad_den)))**2
        timer.reset('Computing intensities', report=verbose)

        py_utils.write_density(intens_fname, intens, binary=True)
        timer.reset('Writing intensities', report=verbose)

        timer.report_time_since_beginning()

def main():
    '''Parses command line arguments to create 3D intensity file using config file parameters'''
    parser = argparse.ArgumentParser(description='Make intensities from density map')
    parser.add_argument('-c', '--config_fname', default='config.ini',
                        help='Path to config file (Default: config.ini)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Say yes to all prompts')
    args = parser.parse_args()

    logging.basicConfig(filename='simdata.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('\n\nStarting.... make_intensities')
    logging.info(' '.join(sys.argv))

    make_intens(args.config_fname, yes=args.yes, verbose=args.verbose)

if __name__ == '__main__':
    main()
