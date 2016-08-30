#!/usr/bin/env python
import numpy as np
import os
import subprocess
import argparse
import logging
import sys
from py_src import py_utils

if __name__ == "__main__":
    logging.basicConfig(filename="recon.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser("Creates new reconstruction instance based on template in this folder")
    parser.add_argument("-Q", "--make_quat_only", dest="make_quat_only", action="store_true", default=False)
    parser.add_argument("-D", "--make_data_only", dest="make_data_only", action="store_true", default=False)
    parser.add_argument("-c", "--config_file", dest="config_file", default="config.ini")
    parser.add_argument("--skip_densities", dest="skip_densities", action="store_true", default=False)
    parser.add_argument("--skip_intensities", dest="skip_intensities", action="store_true", default=False)
    parser.add_argument("--skip_detector", dest="skip_detector", action="store_true", default=False)
    parser.add_argument("--skip_data", dest="skip_data", action="store_true", default=False)
    args = parser.parse_args()
    logging.info("\n\nStarting.... setup")
    logging.info(' '.join(sys.argv))

    if args.make_quat_only:
        args.skip_densities     = True
        args.skip_detector      = True
        args.skip_intensities   = True
        args.skip_data          = True
        args.skip_quat          = False
    elif args.make_data_only:
        args.skip_densities     = True
        args.skip_detector      = True
        args.skip_intensities   = True
        args.skip_data          = False
        args.skip_quat          = True
    else:
        logging.info("Going with the default full workflow")

    # Sequentially step through the simulation workflow
    if not args.skip_densities:
        cmd = "./make_densities.py -c " + args.config_file + " -v"
        logging.info(20*"=" + "\n")
        logging.info(20*"=" + "\n" + cmd)
        subprocess.call(cmd, shell=True)

    if not args.skip_intensities:
        cmd = "./make_intensities.py -c " + args.config_file + " -v"
        logging.info(20*"=" + "\n")
        logging.info(20*"=" + "\n" + cmd)
        subprocess.call(cmd, shell=True)

    if not args.skip_detector:
        cmd = "./make_detector.py -c " + args.config_file + " -v"
        logging.info(20*"=" + "\n")
        logging.info(20*"=" + "\n" + cmd)
        subprocess.call(cmd, shell=True)

    if not args.skip_data:
        cmd = "./make_data -c " + args.config_file
        logging.info(20*"=" + "\n")
        logging.info(20*"=" + "\n" + cmd)
        subprocess.call(cmd, shell=True)
