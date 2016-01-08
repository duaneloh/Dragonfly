#!/usr/bin/env python
import numpy as np
import os
import argparse
import sys
from py_src import py_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates new reconstruction instance based on template in this folder")
    parser.add_argument("-c", "--config_file", dest="config_file", default="config.ini")
    parser.add_argument("--skip_densities", dest="skip_densities", action="store_true", default=False)
    parser.add_argument("--skip_intensities", dest="skip_intensities", action="store_true", default=False)
    parser.add_argument("--skip_detector", dest="skip_detector", action="store_true", default=False)
    parser.add_argument("--skip_data", dest="skip_data", action="store_true", default=False)
    parser.add_argument("--skip_quat", dest="skip_quat", action="store_true", default=False)
    args = parser.parse_args()

    if not args.skip_densities:
        cmd = "./make_densities.py -c " + args.config_file + " -v"
        print 80*"=" + "\n" + cmd
        os.system(cmd)

    if not args.skip_intensities:
        cmd = "./make_intensities.py -c " + args.config_file + " -v"
        print 80*"=" + "\n" + cmd
        os.system(cmd)

    if not args.skip_detector:
        cmd = "./make_detector.py -c " + args.config_file + " -v"
        print 80*"=" + "\n" + cmd
        os.system(cmd)

    if not args.skip_data:
        cmd = "./make_data " + args.config_file
        print 80*"=" + "\n" + cmd
        os.system(cmd)

    if not args.skip_quat:
        py_utils.name_quat_file_sensibly(args.config_file)
        cmd = "./make_quaternion " + args.config_file
        print 80*"=" + "\n" + cmd
        os.system(cmd)
