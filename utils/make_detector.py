#!/usr/bin/env python
import numpy as np
import argparse
import sys
import os
import logging
from py_src import read_config
from py_src import py_utils

if __name__ == "__main__":
    logging.basicConfig(filename="recon.log", level=logging.INFO, format='%(asctime)s - %(levelname)s -%(message)s')
    parser      = py_utils.my_argparser(description="make detector")
    args        = parser.special_parse_args()

    det_file    = os.path.join(args.main_dir, read_config.get_param(args.config_file, 'make_detector', "out_detector_file"))
    to_write    = py_utils.check_to_overwrite(det_file)
    logging.info("Starting make_detector....")
    logging.info(' '.join(sys.argv))

    if to_write:
        timer       = py_utils.my_timer()
        pm          = read_config.get_detector_config(args.config_file, show=args.vb)
        q_pm        = read_config.compute_q_params(pm['detd'], pm['detsize'], pm['pixsize'], pm['wavelength'], show=args.vb)
        timer.reset_and_report("Reading experiment parameters") if args.vb else timer.reset()

        fov_len     = int(np.ceil(q_pm['fov_in_A']/q_pm['half_p_res']) + 1)
        det_cen     = (pm['detsize']-1)/2.
        qscaling    = 1. / pm['wavelength'] / q_pm['q_sep']
        (x, y)      = np.mgrid[0:pm['detsize'], 0:pm['detsize']]
        (x, y)      = (x.flatten()-det_cen, y.flatten()-det_cen)
        (qx, qy)    = (pm['pixsize']*x, pm['pixsize']*y)
        norm        = np.sqrt(qx*qx + qy*qy + pm['detd']*pm['detd'])
        polar       = read_config.compute_polarization(pm['polarization'], qx, qy, norm)
        (qx, qy)    = (qx*qscaling/norm, qy*qscaling/norm)
        qz          = qscaling*(pm['detd']/norm - 1.)
        solid_angle = pm['detd']*(pm['pixsize']*pm['pixsize']) / np.power(norm, 3.0)
        solid_angle = polar*solid_angle
        val_zero    = np.zeros_like(solid_angle)
        val_one     = np.ones_like(solid_angle)
        r           = np.sqrt(x*x + y*y)
        mask        = np.where(r>det_cen, val_one, val_zero)
        mask        = np.where(r<pm['stoprad'], 2*val_one, mask).astype('int')
        timer.reset_and_report("Creating detector") if args.vb else timer.reset()

        with open(det_file, "w") as fp:
            fp.write(str(pm['detsize']*pm['detsize']) + "\n")
            for t0,t1,t2,t3,t4 in zip(qx,qy,qz,solid_angle,mask):
                #txt = "{:21.15e} {:21.15e} {:21.15e} {:21.15e} {:d}\n".format(t0, t1, t2, t3, t4)
                txt = "%21.15e %21.15e %21.15e %21.15e %d\n"%(t0, t1, t2, t3, t4)
                fp.write(txt)
        timer.reset_and_report("Writing detector") if args.vb else timer.reset()

        timer.report_time_since_beginning() if args.vb else timer.reset()
    else:
        pass
