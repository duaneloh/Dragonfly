#!/usr/bin/env python

import numpy as np
import sys
import os
from py_src import py_utils
from py_src import read_config

if __name__ == "__main__":
    # Read detector and photons file from config
    parser      = py_utils.my_argparser(description="make detector")
    args        = parser.special_parse_args()

    det_file    = os.path.join(args.main_dir, read_config.get_param(args.config_file, 'make_detector', "out_detector_file"))
    photons_file    = os.path.join(args.main_dir, read_config.get_param(args.config_file, 'make_data', "out_photons_file"))

    # Parse detector file
    qx, qy, qz = np.loadtxt(det_file, skiprows=1, usecols=(0,1,2), unpack=True)

    # Calculate detd for each pixel
    # Set q=0 pixel detd to mean of the rest
    qz[qz==0.] = 1.
    detd = -(qx**2 + qy**2 + qz**2) / (2.*qz)
    qz[qz==1.] = 0.
    detd[qz==0.] = np.mean(detd[qz!=0.])

    # Calculate x and y for each pixel
    x = np.round(qx*detd/(detd+qz)).astype('i4')
    y = np.round(qy*detd/(detd+qz)).astype('i4')
    x -= x.min()
    y -= y.min()

    # Read photon data
    with open(photons_file, 'rb') as f:
        num_data = np.fromfile(f, dtype='i4', count=1)[0]
        num_pix = np.fromfile(f, dtype='i4', count=1)[0]
        print photons_file+ ': num_data = %d, num_pix = %d' % (num_data, num_pix)
        if num_pix != len(x):
            print 'Detector and photons file dont agree on num_pix'
        f.seek(1024, 0)
        ones = np.fromfile(f, dtype='i4', count=num_data)
        multi = np.fromfile(f, dtype='i4', count=num_data)
        place_ones = np.fromfile(f, dtype='i4', count=ones.sum())
        place_multi = np.fromfile(f, dtype='i4', count=multi.sum())
        count_multi = np.fromfile(f, dtype='i4', count=multi.sum())

    # Place photons in powder array
    powder = np.zeros((x.max()+1,y.max()+1))

    np.add.at(powder, (x[place_ones], y[place_ones]), 1)
    np.add.at(powder, (x[place_multi], y[place_multi]), count_multi)

    # Write float64 array to file
    powder.tofile('data/powder.bin')
