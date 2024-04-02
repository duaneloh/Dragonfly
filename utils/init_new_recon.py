#!/usr/bin/env python

'''Initialize new reconstruction directory:
    This script creates the directory structure used by Dragonfly and
    copies or symlinks all relevant programs into the folder.
    Use init_new_recon.py -h for additional options.
'''

from __future__ import print_function
import os
import os.path as op
import sys
import shutil
import argparse
import logging

def _name_recon_dir(tag, num):
    return "%s_%04d"%(tag, num)

def create_new_recon_dir(tag="recon", num=1, prefix="./"):
    '''Create reconstruction directory
    For given tag, creates directory with first number which does not already exist
    'prefix' option can be set if parent folder is not the current directory
    '''
    recon_dir = op.join(prefix, _name_recon_dir(tag, num))
    while op.exists(recon_dir):
        num += 1
        recon_dir = op.join(prefix, op.join(_name_recon_dir(tag, num)))
    logging.info('New recon directory created with name: %s', recon_dir)
    os.mkdir(recon_dir)
    os.mkdir(op.join(recon_dir, 'data'))
    os.mkdir(op.join(recon_dir, 'images'))
    os.mkdir(op.join(recon_dir, 'logs'))
    if not op.exists(_name_recon_dir(tag, num)):
        os.symlink(recon_dir, _name_recon_dir(tag, num))
    return recon_dir

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
    args.recon_prefix = op.realpath(args.recon_prefix)

    curr_dir = os.getcwd()
    parent_dir = op.realpath(op.join(op.dirname(op.realpath(sys.argv[0])), os.pardir))
    new_recon_dir = create_new_recon_dir(tag=args.recon_tag, num=args.run_tag, prefix=args.recon_prefix)
    print(80*"=")
    print("Initializing new directory and creating soft links to useful utilities.")
    print("Type 'dragonfly_init -h' for options")
    print("See http://github.com/duaneloh/Dragonfly/wiki/FAQ for troubleshooting tips.")
    print(80*"=")

    if args.recon_prefix != './':
        print("Created new directory:", new_recon_dir)

    os.chdir(new_recon_dir)

    src = op.join(parent_dir, 'aux')
    os.symlink(src, 'aux')
    src = op.join(parent_dir, 'utils')
    os.symlink(src, 'utils')
    src = op.join(parent_dir, 'bin/emc')
    os.symlink(src, 'emc')
    src = op.join(parent_dir, 'config.ini')
    shutil.copy(src, 'config.ini')

    os.chdir(curr_dir)

if __name__ == "__main__":
    main()
