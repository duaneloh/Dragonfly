#!/usr/bin/env python

'''Module to make simulated data'''

import sys
import subprocess
import argparse
import logging

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
    parser.add_argument("-y", "--yes",
                        action="store_true", default=False)
    parser.add_argument("--skip_densities", dest="skip_densities",
                        action="store_true", default=False)
    parser.add_argument("--skip_intensities", dest="skip_intensities",
                        action="store_true", default=False)
    parser.add_argument("--skip_detector", dest="skip_detector",
                        action="store_true", default=False)
    parser.add_argument("--skip_data", dest="skip_data",
                        action="store_true", default=False)
    args = parser.parse_args()
    logging.info("\n\nStarting.... setup")
    logging.info(' '.join(sys.argv))

    # Sequentially step through the simulation workflow
    if not args.skip_densities:
        cmd = "./make_densities.py -c " + args.config_file + " -v"
        if args.yes:
            cmd += " -y"
        logging.info(20*"=" + "\n")
        logging.info(20*"=" + "\n" + cmd)
        subprocess.call(cmd, shell=True)

    if not args.skip_intensities:
        cmd = "./make_intensities.py -c " + args.config_file + " -v"
        if args.yes:
            cmd += " -y"
        logging.info(20*"=" + "\n")
        logging.info(20*"=" + "\n" + cmd)
        subprocess.call(cmd, shell=True)

    if not args.skip_detector:
        cmd = "./make_detector.py -c " + args.config_file + " -v"
        if args.yes:
            cmd += " -y"
        logging.info(20*"=" + "\n")
        logging.info(20*"=" + "\n" + cmd)
        subprocess.call(cmd, shell=True)

    if not args.skip_data:
        cmd = "./make_data -c " + args.config_file
        logging.info(20*"=" + "\n")
        logging.info(20*"=" + "\n" + cmd)
        subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    main()
