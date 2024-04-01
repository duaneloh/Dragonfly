#!/usr/bin/env python

'''Module to make simulated data'''

import sys
import os
import subprocess
import argparse
import logging

from dragonfly.utils import make_densities, make_intensities
from dragonfly.utils import make_detector, make_data

def main():
    '''Runs through simulation utilities to generate data
    Utilities:
        make_densities
        make_intensities
        make_detector
        make_data
    '''
    logging.basicConfig(filename="recon.log", level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Generates simulated data using standard pipeline')
    parser.add_argument("-c", "--config_file",
                        dest="config_file", default="config.ini")
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("--skip_densities", action="store_true")
    parser.add_argument("--skip_intensities", action="store_true")
    parser.add_argument("--skip_detector", action="store_true")
    parser.add_argument("--skip_data", action="store_true")
    args = parser.parse_args()
    logging.info("\n\nStarting.... setup")
    logging.info(' '.join(sys.argv))

    base_dir = os.path.realpath(os.path.dirname(args.config_file))
    print(base_dir)
    curr_dir = os.getcwd()
    os.chdir(base_dir)

    # Sequentially step through the simulation workflow
    if not args.skip_densities:
        print('make_densities...')
        make_densities.make_dens(args.config_file, yes=args.yes, verbose=True)

    if not args.skip_intensities:
        print('make_intensities...')
        make_intensities.make_intens(args.config_file, yes=args.yes, verbose=True)

    if not args.skip_detector:
        print('make_detector...')
        make_detector.make_detector(args.config_file, yes=args.yes, verbose=True)

    if not args.skip_data:
        print('make_data...')
        make_data.make_data(args.config_file, yes=args.yes, verbose=True)

    os.chdir(curr_dir)

if __name__ == "__main__":
    main()
