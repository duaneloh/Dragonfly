#!/usr/bin/env python

'''Module to make simulated data'''

import sys
import os
import subprocess
import argparse
import logging

from . import make_densities, make_intensities, make_detector

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
    parser = argparse.ArgumentParser('Creates new reconstruction instance'\
                                     'based on template in this folder')
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
        make_densities.make_dens(args.config_file, yes=args.yes, verbose=True)

    if not args.skip_intensities:
        make_intensities.make_intens(args.config_file, yes=args.yes, verbose=True)

    if not args.skip_detector:
        make_detector.make_detector(args.config_file, yes=args.yes, verbose=True)

    if not args.skip_data:
        cmd = "utils/make_data -c " + args.config_file 
        cmd += '-y' if args.yes else ''
        logging.info(20*"=" + "\n")
        logging.info(20*"=" + "\n" + cmd)
        subprocess.call(cmd.split())

    os.chdir(curr_dir)

if __name__ == "__main__":
    main()
