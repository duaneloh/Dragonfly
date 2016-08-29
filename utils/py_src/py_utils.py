import numpy as np
import time
import ConfigParser
import argparse
import os
import sys
import re
import logging
from glob import glob

class my_timer(object):
    def __init__(self):
        self.t0 = time.time()
        self.ts = self.t0

    def reset(self):
        t1 = time.time()
        self.t0 = t1

    def reset_global(self):
        t1 = time.time()
        self.ts = t1

    def reset_and_report(self, msg):
        t1 = time.time()
        #print "{:-<30}:{:5.5f} seconds".format(msg, t1-self.t0)
        logging.info("{:-<30}:{:5.5f} seconds".format(msg, t1-self.t0))
        self.t0 = t1

    def report_time_since_beginning(self):
        logging.info("="*20)
        logging.info("{:-<30}:{:5.5f} seconds".format("Since beginning", time.time() - self.ts))

class my_argparser(argparse.ArgumentParser):
    def __init__(self, description=""):
        argparse.ArgumentParser.__init__(self, description=description)
        self.add_argument("-c", "--config_file", dest="config_file",
                          help="config file (defaults to config.ini)")
        self.add_argument("-v", "--verbose", dest="vb", action="store_true", default=False)
        self.add_argument("-m", "--main_dir", dest="main_dir",
                          help="relative path to main repository directory\n(where data aux utils are stored)")

    def special_parse_args(self):
        args = self.parse_args()
        if not args.config_file:
            args.config_file = "config.ini"
            logging.info("Config file not specified. Using " + args.config_file)
        if not args.main_dir:
            args.main_dir = os.path.split(os.path.abspath(args.config_file))[0]
            logging.info("Main directory not specified. Using " + args.main_dir)
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
        l       = int(np.round(np.power(sz, 1./2.)))
        out_den = den.reshape(l,l,l)
    return out_den

def check_to_overwrite(fn):
    overwrite = True
    yes = set(['yes', 'y', '', 'yup', 'ya'])
    no  = set(['no', 'n', 'nope', 'nay', 'not'])
    if os.path.isfile(fn):
        sys.stdout.write("%s is present. Overwrite? [Y or Return/N]: " % fn)
        choice = raw_input().lower()
        if choice in yes:
            overwrite = True
            print "Overwriting " + fn
            logging.info("Overwriting " + fn)
        elif choice in no:
            overwrite = False
            print "Not overwriting " + fn
            logging.info("Not overwriting " + fn)
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'")
            overwrite = False
    return overwrite

def confirm_oversampling(ratio):
    proceed = True
    done = False
    yes = set(['yes', 'y', '', 'yup', 'ya'])
    no  = set(['no', 'n', 'nope', 'nay', 'not'])
    print 'Oversampling ratio = %.2f is a little high. This is inefficient.' % ratio
    print 'Please see http://www.github.com/duaneloh/Dragonfly/wiki/Oversampling for tips'
    sys.stdout.write('Continue anyway? [Y or Return/N]: ')
    while not done:
        choice = raw_input().lower()
        if choice in yes:
            proceed = True
            done = True
            logging.info("Continuing with minimum oversampling ratio = %.2f" % ratio)
        elif choice in no:
            proceed = False
            done = True
            logging.info("Minimum oversampling ratio of %.2f too high. Quitting" % ratio)
        else:
            sys.stdout.write("Please respond with 'yes' or 'no': ")
            proceed = False
    return proceed 

def name_recon_dir(tag, num):
    return "%s_%04d"%(tag, num)

def create_new_recon_dir(tag="recon", num=1, prefix="./"):
    recon_dir = os.path.join(prefix, name_recon_dir(tag, num))
    while(os.path.exists(recon_dir)):
        num += 1
        recon_dir = os.path.join(prefix, os.path.join(name_recon_dir(tag, num)))
    msg = 'New recon directory created with name: ' + recon_dir
    logging.info(msg)
    os.mkdir(recon_dir)
    sub_dir = {"data":["scale", "orientations", "mutualInfo", "weights", "output"], "images":[]}
    for k,v in sub_dir.items():
        os.mkdir(os.path.join(recon_dir, k))
        if len(v) > 0:
            for vv in v:
                os.mkdir(os.path.join(recon_dir, k, vv))
    if not os.path.exists(name_recon_dir(tag, num)):
        os.symlink(recon_dir, name_recon_dir(tag, num))
    return recon_dir

def use_last_recon_as_starting_model(config_fname, output_subdir="output"):
    config = ConfigParser.ConfigParser()
    config.read(config_fname)

    emc_data_dir = os.path.join(config.get("emc", "out_folder"), output_subdir)
    recon_out_files = glob(os.path.join(emc_data_dir, "intens*.bin"))
    (max_tag, max_file) = (0, "")
    for f in recon_out_files:
        t = int(re.search("intens_(\d+).bin", f).group(1))
        if t > max_tag:
            max_tag = t
            max_file = f
    logging.info("Setting start_model_file to " + max_file)
    config.set("emc", "start_model_file", max_file)
    with open("config.ini", "w") as fp:
        config.write(fp)

def increment_quat_file_sensibly(config_fname, incr):
    config = ConfigParser.ConfigParser()
    config.read(config_fname)

    quat_num_div = int(config.get("emc", "num_div"))
    msg = ["Setting quaternion from", str(quat_num_div), "to", str(quat_num_div+incr)]
    logging.info(' '.join(msg))
    quat_num_div += incr
    config.set("emc", "num_div", quat_num_div)

    with open("config.ini", "w") as fp:
        config.write(fp)
