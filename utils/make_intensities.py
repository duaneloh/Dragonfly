#!/usr/bin/env python
import numpy as np
import argparse
import sys
import os
import logging
from py_src import read_config
from py_src import py_utils

if __name__ == "__main__":
    logging.basicConfig(filename="recon.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser      = py_utils.my_argparser(description="make intensities")
    args        = parser.special_parse_args()

    den_file    = os.path.join(args.main_dir, read_config.get_filename(args.config_file, 'make_intensities', "in_density_file"))
    intens_file = os.path.join(args.main_dir, read_config.get_filename(args.config_file, 'make_intensities', "out_intensity_file"))
    to_write    = py_utils.check_to_overwrite(intens_file)
    logging.info("\n\nStarting.... make_intensities")
    logging.info(' '.join(sys.argv))

    if to_write:
        timer       = py_utils.my_timer()
        pm          = read_config.get_detector_config(args.config_file, show=args.vb)
        q_pm        = read_config.compute_q_params(pm['detd'], pm['dets_x'], pm['dets_y'], pm['pixsize'], pm['wavelength'], show=args.vb)
        timer.reset_and_report("Reading experiment parameters") if args.vb else timer.reset()

        fov_len     = 2 * int(np.ceil(q_pm['fov_in_A']/q_pm['half_p_res']/2.)) + 3
        logging.info('Volume size: %d' % fov_len) 
        den         = py_utils.read_density(den_file, binary=True)
        min_over    = float(fov_len)/den.shape[0]
        if min_over > 12:
            if py_utils.confirm_oversampling(min_over) is False:
                sys.exit(0)
        timer.reset_and_report("Reading densities") if args.vb else timer.reset()

        pad_den     = np.zeros(3*(fov_len,))
        den_sh      = den.shape
        pad_den[:den_sh[0],:den_sh[1],:den_sh[2]] = den.copy()
        ft          = np.abs(np.fft.fftshift(np.fft.fftn(pad_den)))
        intens      = ft*ft
        timer.reset_and_report("Computing intensities") if args.vb else timer.reset()

        py_utils.write_density(intens_file, intens, binary=True)
        timer.reset_and_report("Writing intensities") if args.vb else timer.reset()

        timer.report_time_since_beginning() if args.vb else timer.reset()
    else:
        pass
