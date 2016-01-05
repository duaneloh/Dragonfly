import numpy as np
import argparse
import sys
import os
from py_src import read_config
from py_src import py_utils

if __name__ == "__main__":

    timer       = py_utils.my_timer()
    parser      = argparse.ArgumentParser(description="make intensities")
    parser.add_argument(dest='config_file')
    parser.add_argument("-v", "--verbose", dest="vb", action="store_true", default=False)
    parser.add_argument("-m", "--main_dir", dest="main_dir", help="relative path to main repository directory\n(where data aux utils are stored)")
    args        = parser.parse_args()
    args.main_dir = args.main_dir if args.main_dir else os.path.dirname(args.config_file)

    pm          = read_config.get_detector_config(args.config_file, show=args.vb)
    q_pm        = read_config.compute_q_params(pm['detd'], pm['detsize'], pm['pixsize'], pm['wavelength'], show=args.vb)
    timer.reset_and_report("Reading experiment parameters") if args.vb else timer.reset()

    fov_len     = int(np.ceil(q_pm['fov_in_A']/q_pm['half_p_res']) + 1)
    den_file    = os.path.join(args.main_dir, read_config.get_param(args.config_file, 'make_intensities', "in_density_file"))
    intens_file = os.path.join(args.main_dir, read_config.get_param(args.config_file, 'make_intensities', "out_intensity_file"))
    den         = py_utils.read_density(den_file, binary=True)
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
