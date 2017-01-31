#!/usr/bin/env python
import numpy as np
import argparse
import sys
import os
import logging
from py_src import read_config
from py_src import py_utils

if __name__ == "__main__":
    logging.basicConfig(filename="recon.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser      = py_utils.my_argparser(description="make detector")
    args        = parser.special_parse_args()

    det_file    = os.path.join(args.main_dir, read_config.get_filename(args.config_file, 'make_detector', "out_detector_file"))
    to_write    = py_utils.check_to_overwrite(det_file)
    logging.info("\n\nStarting make_detector....")
    logging.info(' '.join(sys.argv))

    if to_write:
        timer       = py_utils.my_timer()
        pm          = read_config.get_detector_config(args.config_file, show=args.vb)
        q_pm        = read_config.compute_q_params(pm['detd'], pm['dets_x'], pm['dets_y'], pm['pixsize'], pm['wavelength'], pm['ewald_rad'], show=args.vb)
        timer.reset_and_report("Reading experiment parameters") if args.vb else timer.reset()

        fov_len     = 2 * int(np.ceil(q_pm['fov_in_A']/q_pm['half_p_res']/2.)) + 3
        det_cen_x   = pm['detc_x']
        det_cen_y   = pm['detc_y']
        qscaling    = 1. / pm['wavelength'] / q_pm['q_sep']
        (x, y)      = np.mgrid[0:pm['dets_x'], 0:pm['dets_y']]
        (x, y)      = (x.flatten()-det_cen_x, y.flatten()-det_cen_y)
        (px, py)    = (pm['pixsize']*x, pm['pixsize']*y)
        norm        = np.sqrt(px*px + py*py + pm['detd']*pm['detd'])
        polar       = read_config.compute_polarization(pm['polarization'], px, py, norm)
        (qx, qy)    = (px*qscaling/norm, py*qscaling/norm)
        qz          = qscaling*(pm['detd']/norm - 1.)
        logging.info('{:<15}:{:10.4f}'.format('qmax', np.sqrt(qx*qx + qy*qy + qz*qz).max()))
        solid_angle = pm['detd']*(pm['pixsize']*pm['pixsize']) / np.power(norm, 3.0)
        solid_angle = polar*solid_angle
        val_zero    = np.zeros_like(solid_angle)
        val_one     = np.ones_like(solid_angle)
        r           = np.sqrt(x*x + y*y)
        if pm['mask_fname'] == None:
            mask        = np.where(r>min(det_cen_x, det_cen_y), val_one, val_zero)
            mask        = np.where(r<pm['stoprad'], 2*val_one, mask).astype('int')
        else:
            mask = np.fromfile(pm['mask_fname'], '=u1')
            mask[(r>min(det_cen_x, det_cen_y)) & (mask==0)] = 1
        timer.reset_and_report("Creating detector") if args.vb else timer.reset()

        with open(det_file, "w") as fp:
            fp.write(str(pm['dets_x']*pm['dets_y']) + "\n")
            for t0,t1,t2,t3,t4 in zip(qx,qy,qz,solid_angle,mask):
                #txt = "{:21.15e} {:21.15e} {:21.15e} {:21.15e} {:d}\n".format(t0, t1, t2, t3, t4)
                txt = "%21.15e %21.15e %21.15e %21.15e %d\n"%(t0, t1, t2, t3, t4)
                fp.write(txt)
        timer.reset_and_report("Writing detector") if args.vb else timer.reset()

        timer.report_time_since_beginning() if args.vb else timer.reset()
    else:
        pass
