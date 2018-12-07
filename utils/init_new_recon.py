#!/usr/bin/env python

'''Initialize new reconstruction directory:
    This script creates the directory structure used by Dragonfly and
    copies or symlinks all relevant programs into the folder.
    Use init_new_recon.py -h for additional options.
'''

from __future__ import print_function
import os
import sys
import shutil
import argparse
sys.path.append(os.path.dirname(__file__))
from py_src import py_utils # pylint: disable=import-error

def main():
    '''Parses command line arguments and creates new reconstruction directory'''
    parser = argparse.ArgumentParser('Creates new reconstruction instance'\
                                     'based on template in this folder')
    parser.add_argument("-t", "--recon_file_tag", dest="recon_tag", default="recon",
                        help="prefixes the reconstruction folders with your specified tags.")
    parser.add_argument("-r", "--recon_run_num", dest="run_tag", type=int, default=1,
                        help="give your reconstruction a specific number if it doesn't already exist")
    parser.add_argument("-p", "--recon_prefix", dest="recon_prefix", default="./",
                        help="path to the folder containing the reconstruction folder")
    args = parser.parse_args()
    args.recon_prefix = os.path.realpath(args.recon_prefix)

    curr_dir = os.getcwd()
    parent_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), os.pardir))
    new_recon_dir = py_utils.create_new_recon_dir(tag=args.recon_tag, num=args.run_tag,
                                                  prefix=args.recon_prefix)
    print(80*"=")
    print("Initializing new directory and creating soft links to useful utilities.")
    print("Type 'dragonfly_init -h' for options")
    print("See http://github.com/duaneloh/Dragonfly/wiki/FAQ for troubleshooting tips.")
    print(80*"=")

    if args.recon_prefix != './':
        print("Created new directory:", new_recon_dir)

    os.chdir(new_recon_dir)

    src = os.path.join(parent_dir, 'aux')
    os.symlink(src, 'aux')
    src = os.path.join(parent_dir, 'utils')
    os.symlink(src, 'utils')
    src = os.path.join(parent_dir, 'bin/emc')
    os.symlink(src, 'emc')
    src = os.path.join(parent_dir, 'config.ini')
    shutil.copy(src, 'config.ini')

    os.chdir(curr_dir)

if __name__ == "__main__":
    main()
