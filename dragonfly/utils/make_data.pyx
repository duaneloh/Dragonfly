import sys
import os.path as op
import time
import argparse
import logging

import numpy as np
import dragonfly
from cython.parallel import parallel, prange
cimport openmp

from .py_src import read_config, py_utils

def rand_quat():
    while True:
        qvals = np.random.random(4) - 0.5
        qnorm = np.linalg.norm(qvals)
        if qnorm < 0.25:
            break
    return qvals / qnorm

def make_data(config_fname, yes=False, verbose=False):
    config = read_config.MyConfigParser()
    config.read(config_fname)

    num_data = config.getint('make_data', 'num_data')
    fluence = config.getfloat('make_data', 'fluence')
    intens_fname = config.get_filename('make_data', 'in_intensity_file')
    det_fname = config.get_filename('make_data', 'in_detector_file')
    out_fname = config.get_filename('make_data', 'out_photons_file')

    if not (yes or py_utils.check_to_overwrite(out_fname)):
        return

    timer = py_utils.MyTimer()
    model = dragonfly.Model(1)
    try:
        model.allocate(intens_fname)
    except ValueError as err:
        size = err.args[1]
        model.free()
        model = dragonfly.Model(size)
        model.allocate(intens_fname)

    det = dragonfly.CDetector(det_fname, norm=False)

    rescale = fluence * 2.81794e-9**2
    model.model1[0] *= rescale

    hdf5_output = True
    if op.splitext(out_fname)[1] == '.emc':
        hdf5_output = False
    timer.reset('Ready to generate data', report=verbose)

    stime = time.time()
    wemc = dragonfly.EMCWriter(out_fname, det.num_pix, hdf5=hdf5_output)
    view = np.zeros(det.num_pix, dtype='f8')
    #print(openmp.omp_get_max_threads())

    for i in range(num_data):
        phot = np.random.poisson(model.slice_gen(rand_quat(), det, view=view))
        phot[det.raw_mask == 2] = 0
        wemc.write_frame(phot.ravel())
        sys.stderr.write('\r%d/%d'%(i+1, num_data))
    sys.stderr.write('\n')
    timer.reset('Generated frames', report=verbose)

    wemc.finish_write()
    print('Time taken: %.3f s' % (time.time()-timer._time_start))
    timer.reset('Writing to file', report=verbose)

    timer.report_time_since_beginning()

def main():
    '''Parse command line arguments and generate electron density volume with config file'''
    parser = argparse.ArgumentParser(description='Make data file from 3D intensities')
    parser.add_argument('-c', '--config_fname', default='config.ini',
                        help='Path to config file (Default: config.ini)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Say yes to all prompts')
    args = parser.parse_args()

    logging.basicConfig(filename='simdata.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('\n\nStarting.... make_data')
    logging.info(' '.join(sys.argv))

    make_data(args.config_fname, yes=args.yes, verbose=args.verbose)

if __name__ == '__main__':
    main()
