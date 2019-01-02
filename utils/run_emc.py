#!/usr/bin/env python

'''Module to wrap running of EMC command'''

from __future__ import print_function
import subprocess
import argparse
import logging
import sys
from py_src import py_utils

def _apply_presets(args):
    # Here are some custom hybrid configurations
    if args.kane:
        args.num_mpi = 9
        args.num_threads = 8
    elif args.kahuna:
        args.num_mpi = 15
        args.num_threads = 6
    elif args.bayes:
        args.num_mpi = 4
        args.num_threads = 12
    elif args.tukey:
        args.num_mpi = 6
        args.num_threads = 4
    elif args.slac:
        args.num_mpi = 16
        args.num_threads = 2
    elif args.davinci:
        args.num_mpi = 3
        args.num_threads = 4

def _calculate_openmp_cmd(args):
    # We might not need this anymore, except with the extend with quaternion up-refinement.
    # Decide if we are just refining the reconstruction with more iterations
    if args.resume_recon:
        ext_str = "-r"
    elif args.resume_recon_add_quat:
        args.quat_add = 1
        ext_str = "-r"
    elif args.resume_recon_increase_beta:
        args.beta_incr = 2
        ext_str = "-r"
    else:
        ext_str = ""

    if args.num_threads == -1:
        cmd = ["./emc", "-c", args.config_file, ext_str, str(args.num_iter)]
    else:
        cmd = ("./emc -c %s -t %d %s %d"%(args.config_file, args.num_threads,
                                          ext_str, args.num_iter)).split()
    return cmd

def main():
    '''Uses command line arguments to wrap and run the emc program
    Run run_emc.py -h for a list of options
    '''
    # logging config must occur before MyArgparser, because latter already starts logging
    logging.basicConfig(filename="recon.log", level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser("Starts EMC reconstruction")
    parser.add_argument("-c", "--config_file", dest="config_file", default="config.ini")
    parser.add_argument("-r", dest="resume_recon", action='store_true', default=False,
                        help="resume reconstruction from last output")
    parser.add_argument("-R", dest="resume_recon_add_quat", action='store_true', default=False,
                        help="same as -r, except we increase quaternion sampling by one")
    parser.add_argument("-q", dest="quat_add", type=int, default=0,
                        help="increase quaternion sampling by an integer (default=0)")
    parser.add_argument("-B", dest="resume_recon_increase_beta", action='store_true', default=False,
                        help="same as -r, except we increase beta by a factor of two")
    parser.add_argument("-b", dest="beta_incr", type=float, default=1.,
                        help="increase beta by multiplying with a float (default=1.)")
    parser.add_argument("-m", dest="num_mpi", type=int, default=0,
                        help="number of mpi processes (default=0)")
    parser.add_argument("-i", dest="num_iter", type=int, default=10,
                        help="number of iterations (default=10)")
    parser.add_argument("-t", dest="num_threads", type=int, default=-1,
                        help="number of openMP thread(defaults to $OMP_NUM_THREADS on machine")
    parser.add_argument("--kane", action='store_true', default=False)
    parser.add_argument("--kahuna", action='store_true', default=False)
    parser.add_argument("--bayes", action='store_true', default=False)
    parser.add_argument("--tukey", action='store_true', default=False)
    parser.add_argument("--slac", action='store_true', default=False)
    parser.add_argument("--davinci", action="store_true", default=False)
    parser.add_argument("--dry_run", action='store_true', default=False,
                        help="print commands to screen but we won't actually run them")
    args = parser.parse_args()
    logging.info("\n\nStarting run_emc....")
    logging.info(sys.argv)

    _apply_presets(args)

    openmp_cmd = _calculate_openmp_cmd(args)

    # Determine if quaternions should be incremented in the config file
    if args.quat_add != 0:
        py_utils.increment_quat_file_sensibly(args.config_file, args.quat_add)

    # Beta should incremented in the log file
    if args.beta_incr != 1:
        py_utils.increment_beta_sensibly(args.config_file, args.beta_incr)

    # MPI options
    MPI_options = []
    if args.davinci:
        MPI_options.append("--bind-to none")

    # Switch between openMP only or MPI + openMP
    if args.num_mpi > 0:
        cmd = ' '.join(["mpirun -n", str(args.num_mpi)] + MPI_options + openmp_cmd)
        if not args.dry_run:
            logging.info(20*"=" + "\n")
            logging.info(20*"=" + "\n" + cmd)
            subprocess.call(cmd, shell=True)
        else:
            print(cmd)
    else:
        cmd = ' '.join(openmp_cmd)
        if not args.dry_run:
            logging.info(20*"=" + "\n")
            logging.info(20*"=" + "\n" + cmd)
            subprocess.call(cmd, shell=True)
        else:
            print(cmd)

if __name__ == "__main__":
    main()
