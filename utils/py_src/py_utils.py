import numpy as np
import time
import argparse
import os
import sys

class my_timer(object):
    def __init__(self):
        self.t0 = time.time()
        self.ts = self.t0

    def reset(self):
        t1 = time.time()
        self.t0 = t1

    def reset_and_report(self, msg):
        t1 = time.time()
        print "{:-<30}:{:5.5f} seconds".format(msg, t1-self.t0)
        self.t0 = t1

    def report_time_since_beginning(self):
        print "="*80
        print "{:-<30}:{:5.5f} seconds".format("Since beginning", time.time() - self.ts)

class my_argparser(argparse.ArgumentParser):
    def __init__(self, description=""):
        argparse.ArgumentParser.__init__(self, description=description)
        self.add_argument("-c", "--config_file", dest="config_file")
        self.add_argument("-v", "--verbose", dest="vb", action="store_true", default=False)
        self.add_argument("-m", "--main_dir", dest="main_dir", help="relative path to main repository directory\n(where data aux utils are stored)")

    def special_parse_args(self):
        args = self.parse_args()
        if not args.config_file:
            args.config_file = "../config.ini"
            print "Config file not specified. Using " + args.config_file
        if not args.main_dir:
            args.main_dir = os.path.split(os.path.abspath(args.config_file))[0]
            print "Main directory not specified. Using " + args.main_dir
        return args

def write_density(in_den_file, in_den, binary=True):
    if binary:
        in_den.astype('float64').tofile(in_den_file)
    else:
        with open(in_den_file, "w") as fp:
            for l0 in in_den:
                for l1 in l0:
                    tmp = ' '.join(l1.astype('str'))
                    fp.write(tmp + '\n')

def read_density(in_den_file, binary=True):
    if binary:
        den     = np.fromfile(in_den_file, dtype="float64")
        sz      = len(den)
        l       = int(np.round(np.power(sz, 1./3.)))
        out_den = den.reshape(l,l,l)
    else:
        with open(in_den_file, "r") as fp:
            lines   = fp.readlines()
            den     = []
            for l in lines:
                den.append([float(s) for s in l.strip().split()])
            den     = np.array(den)
        sz      = len(den)
        l       = int(np.power(sz, 1./3.))
        out_den = den.reshape(l,l,l)
    return out_den

def check_to_overwrite(fn):
    overwrite = True
    yes = set(['yes', 'y', '', 'yup', 'ya'])
    no  = set(['no', 'n', 'nope', 'nay', 'not'])
    if os.path.isfile(fn):
        print fn + " is present. Overwrite? [Y/N]"
        choice = raw_input().lower()
        if choice in yes:
            overwrite = True
            print "Overwriting " + fn
        elif choice in no:
            overwrite = False
            print "Not overwriting " + fn
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'")
            overwrite = False
    return overwrite
