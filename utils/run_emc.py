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
    parser.add_argument("-x", dest="auto_extend_recon", action='store_true', default=False,
                        help="continue reconstruction from last output")
    parser.add_argument("-X", dest="auto_extend_recon_add_quat", action='store_true', default=False,
                        help="same as -x, except we increase quaternion sampling by one")
    parser.add_argument("-q", dest="quat_add", type=int, default=0,
                        help="increase quaternion sampling by an integer (default=0)")
    parser.add_argument("-m", dest="num_mpi", type=int, default=0,
                        help="number of mpi processes (default=0)")
    parser.add_argument("-i", dest="num_iter", type=int, default=10,
                        help="number of iterations (default=10)")
    parser.add_argument("-t", dest="num_threads", type=int, default=-1,
                        help="number of openMP thread(defaults to $OMP_NUM_THREADS on machine")
    parser.add_argument("--kahuna", action='store_true', default=False)
    parser.add_argument("--dry_run", action='store_true', default=False,
                        help="print commands to screen but we won't actually run them")
    args = parser.parse_args()
    logging.info("Starting run_emc....")
    logging.info(sys.argv)

    # Here are some custom hybrid configurations
    if args.kahuna:
        args.num_mpi = 7
        args.num_threads = 12

    # We might not need this anymore, except with the extend with quaternion up-refinement.
    # Decide if we are just refining the reconstruction with more iterations
    if not args.dry_run:
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

    # Switch between openMP only or openMPI + openMP
    if args.num_threads == -1:
        openMP_cmd = ["./emc", "-c", str(args.config_file), str(args.num_iter)]
    else:
        openMP_cmd = ["./emc", "-c", str(args.config_file), "-t", str(args.num_threads), str(args.num_iter)]

    if args.num_mpi > 0:
        cmd = ' '.join(["mpirun -n", str(args.num_mpi)] + openMP_cmd)
        if not args.dry_run:
            logging.info(80*"=" + "\n")
            logging.info(80*"=" + "\n" + cmd)
            subprocess.call(cmd, shell=True)
        else:
            print cmd
    else:
        cmd = ' '.join(openMP_cmd)
        if not args.dry_run:
            logging.info(80*"=" + "\n")
            logging.info(80*"=" + "\n" + cmd)
            subprocess.call(cmd, shell=True)
        else:
            print cmd
