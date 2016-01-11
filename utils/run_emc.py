#!/usr/bin/env python
import numpy as np
import os
import subprocess
import argparse
import logging
import sys
from py_src import py_utils

if __name__ == "__main__":
    # logging config must occur before my_argparser, because latter already starts logging
    logging.basicConfig(filename="recon.log", level=logging.INFO, format='%(asctime)s - %(levelname)s -%(message)s')
    parser = argparse.ArgumentParser("Starts EMC reconstruction")
    parser.add_argument("-c", "--config_file", dest="config_file", default="config.ini")
    parser.add_argument("-x", dest="auto_extend_recon", action='store_true', default=False)
    parser.add_argument("-X", dest="auto_extend_recon_add_quat", action='store_true', default=False)
    parser.add_argument("-q", dest="quat_add", type=int, default=0)
    parser.add_argument("-m", dest="num_mpi", type=int, default=0)
    parser.add_argument("-i", dest="num_iter", type=int, default=5)
    parser.add_argument("-t", dest="num_threads", type=int, default=4)
    parser.add_argument("--kahuna", action='store_true', default=False)
    parser.add_argument("--dry_run", action='store_true', default=False)
    args = parser.parse_args()
    logging.info("Starting run_emc....")
    logging.info(sys.argv)

    # Here are some custom hybrid configurations
    if args.kahuna:
        args.num_mpi = 8
        args.num_threads = 12

    # Decide if we are just refining the reconstruction with more iterations
    if args.auto_extend_recon:
        py_utils.use_last_recon_as_starting_model(args.config_file)
    elif args.auto_extend_recon_add_quat:
        args.quat_add = 1
        py_utils.use_last_recon_as_starting_model(args.config_file)

    # Determine of quaternions should be incremented in the log file and recomputed
    if args.quat_add != 0:
        py_utils.increment_quat_file_sensibly(args.config_file, args.quat_add)
        cmd = "./make_quaternion " + args.config_file
        if not args.dry_run:
            logging.info(80*"=" + "\n")
            logging.info(80*"=" + "\n" + cmd)
            subprocess.call(cmd, shell=True)
        else:
            print cmd

    # Switch between openMP only of openMPI + openMP
    if args.num_mpi > 0:
        cmd = ' '.join(["mpirun -n", str(args.num_mpi),
                        "./emc", str(args.num_iter), str(args.config_file), str(args.num_threads)])
        if not args.dry_run:
            logging.info(80*"=" + "\n")
            logging.info(80*"=" + "\n" + cmd)
            subprocess.call(cmd, shell=True)
        else:
            print cmd
    else:
        cmd = ' '.join(["./emc", str(args.num_iter), str(args.config_file), str(args.num_threads)])
        if not args.dry_run:
            logging.info(80*"=" + "\n")
            logging.info(80*"=" + "\n" + cmd)
            subprocess.call(cmd, shell=True)
        else:
            print cmd
