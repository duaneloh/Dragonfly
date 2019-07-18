#!/usr/bin/env python

'''Module to generate detector file from configuration parameters'''

import sys
import os
import logging
import numpy as np
from py_src import read_config
from py_src import py_utils
from py_src import detector

def main():
    '''Parses command line arguments and config file to generate detector file'''
    logging.basicConfig(filename="recon.log", level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = py_utils.MyArgparser(description="make detector")
    args = parser.special_parse_args()

    det_file = os.path.join(args.main_dir,
                            read_config.get_filename(args.config_file,
                                                     'make_detector',
                                                     'out_detector_file'))
    if args.yes:
        to_write = True
    else:
        to_write = py_utils.check_to_overwrite(det_file)
    logging.info("\n\nStarting make_detector....")
    logging.info(' '.join(sys.argv))

    if to_write:
        timer = py_utils.MyTimer()
        pm = read_config.get_detector_config(args.config_file, show=args.vb) # pylint: disable=C0103
        q_pm = read_config.compute_q_params(pm['detd'], pm['dets_x'],
                                            pm['dets_y'], pm['pixsize'],
                                            pm['wavelength'], pm['ewald_rad'], show=args.vb)
        timer.reset_and_report("Reading experiment parameters") if args.vb else timer.reset()

        det = detector.Detector()
        qscaling = 1. / pm['wavelength'] / q_pm['q_sep']
        (x, y) = np.mgrid[0:pm['dets_x'], 0:pm['dets_y']]
        (x, y) = (x.flatten()-pm['detc_x'], y.flatten()-pm['detc_y'])
        norm = np.sqrt(x**2 + y**2 + (pm['detd']/pm['pixsize'])**2)
        (det.qx, det.qy, det.qz) = (x*qscaling/norm, # pylint: disable=C0103
                                    y*qscaling/norm,
                                    qscaling*(pm['detd']/pm['pixsize']/norm - 1.))
        logging.info('%15s:%10.4f', 'qmax', np.sqrt(det.qx**2 + det.qy**2 + det.qz**2).max())

        polar = read_config.compute_polarization(pm['polarization'], x, y, norm)
        det.corr = pm['detd'] / pm['pixsize'] / norm**3
        det.corr *= polar
        rad = np.sqrt(x*x + y*y)
        if pm['mask_fname'] is None:
            det.raw_mask = np.zeros(det.corr.shape, dtype='u1')
            det.raw_mask[rad > min(pm['detc_x'], pm['detc_y'])] = 1
            det.raw_mask[rad < pm['stoprad']] = 2
        else:
            det.raw_mask = np.fromfile(pm['mask_fname'], '=u1')
            det.raw_mask[(rad > min(pm['detc_x'], pm['detc_y'])) & (mask == 0)] = 1
        timer.reset_and_report("Creating detector") if args.vb else timer.reset()

        det.detd = pm['detd'] / pm['pixsize']
        det.ewald_rad = pm['ewald_rad']
        det.write(det_file)
        timer.reset_and_report("Writing detector to %s" % det_file) if args.vb else timer.reset()

        timer.report_time_since_beginning() if args.vb else timer.reset()
    else:
        pass

if __name__ == "__main__":
    main()
