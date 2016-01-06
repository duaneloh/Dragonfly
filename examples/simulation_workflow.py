import numpy as np
import os
import sys

# Python files stored in this directory
sys.path.append("../utils")
from py_src import read_config
from py_src import process_pdb
from py_src import py_utils
import make_densities

if __name__ == "__main__":
    config_file = "config_example.ini"
    main_dir    = "../"
    cmd = "python ../utils/make_densities.py -c " + config_file + " -m " + main_dir + " -v"
    print 80*"=" + "\n" + cmd
    os.system(cmd)

    cmd = "python ../utils/make_intensities.py -c " + config_file + " -m " + main_dir + " -v"
    print 80*"=" + "\n" + cmd
    os.system(cmd)

    cmd = "python ../utils/make_detector.py -c " + config_file + " -m " + main_dir + " -v"
    print 80*"=" + "\n" + cmd
    os.system(cmd)
