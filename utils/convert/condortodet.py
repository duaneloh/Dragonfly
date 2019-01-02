#!/usr/bin/env python

'''
Convert CONDOR geometry conventions to detector file

Needs:
    CXI file produced by CONDOR (eg. photons.cxi)

Produces:
    Detector file in output_folder
'''

import os
import numpy as np
import h5py
import sys
import logging
#Add utils directory to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from py_src import py_utils
from py_src import writeemc
from py_src import read_config

try:
    import condor
except ImportError:
    sys.stderr.write('This utility needs CONDOR to run. Unable to import condor module\n')
    sys.exit(1)

if __name__ == '__main__':
    logging.basicConfig(filename='recon.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = py_utils.my_argparser(description='CONDOR to det')
    parser.add_argument('filename', help='CONDOR file', type=str)
    args = parser.special_parse_args()

    logging.info('Starting condortodet...')
    logging.info(' '.join(sys.argv))
    pm = read_config.get_detector_config(args.config_file, show=args.vb)
    output_folder = read_config.get_filename(args.config_file, 'emc', 'output_folder')

    with h5py.File(args.filename,'r') as f:
        condor_dets_x  = f['detector']['nx'][0]
        condor_dets_y  = f['detector']['ny'][0]
        condor_detc_x  = f['detector']['cx'][0]
        condor_detc_y  = f['detector']['cy'][0]
        condor_detd    = f['detector']['distance'][0] # in units of m
        condor_pixsize = f['detector']['pixel_size'][0] # in units of m
        condor_wavelength = f['source/wavelength'][0] # in units of m
        condor_mask = f['entry_1/data_1/mask'][0]
    assert condor_dets_x == pm['dets_x'], "Missmatch between condor file (%f) and config.ini (%f)" %(condor_dets_x, pm['dets_x'])
    assert condor_dets_y == pm['dets_y'], "Missmatch between condor file (%f) and config.ini (%f)" %(condor_dets_y, pm['dets_y'])
    assert condor_detc_x == pm['detc_x'], "Missmatch between condor file (%f) and config.ini (%f)" %(condor_detc_x, pm['detc_x'])
    assert condor_detc_y == pm['detc_y'], "Missmatch between condor file (%f) and config.ini (%f)" %(condor_detc_y, pm['detc_y'])
    assert (condor_detd * 1e3) == pm['detd'], "Missmatch between condor file (%f) and config.ini (%f)" %(condor_detd * 1e3, pm['detd'])
    assert condor_pixsize * 1e3 == pm['pixsize'], "Missmatch between condor file (%f) and config.ini (%f)" %(condor_pixsize * 1e3, pm['pixsize'])
    assert condor_wavelength * 1e10 == pm['wavelength'], "Missmatch between condor file (%f) and config.ini (%f)" %(condor_wavelength * 1e10, pm['wavelength'])

    # CONDOR CALCULATION OF QMAP AND SOLID ANGLE
    # ---------------------------------------------
    det = condor.Detector(condor_detd, condor_pixsize, 
                          cx=condor_detc_x, cy=condor_detc_y,
                          nx=condor_dets_x, ny=condor_dets_y)
    condor_X, condor_Y = det.generate_xypix(cx=det.get_cx_mean_value(), cy=det.get_cy_mean_value())
    condor_R = np.sqrt(condor_X**2 + condor_Y**2)
    condor_qmap = det.generate_qmap(condor_wavelength, cx=det.get_cx_mean_value(), cy=det.get_cy_mean_value(), order='xyz')
    condor_qx = condor_qmap[:,:,0].flatten()
    condor_qy = condor_qmap[:,:,1].flatten()
    condor_qz = condor_qmap[:,:,2].flatten()
    condor_polar = det.calculate_polarization_factors(cx=det.get_cx_mean_value(),
                                                      cy = det.get_cy_mean_value(), 
                                                      polarization='unpolarized').flatten()
    condor_solid_angle = condor_polar * det.get_all_pixel_solid_angles(cx=det.get_cx_mean_value(),
                                                                       cy = det.get_cy_mean_value()).flatten()

    # DRAGONFLY CALCULATION OF QMAP AND SOLID ANGLE
    # ---------------------------------------------
    q_pm = read_config.compute_q_params(pm['detd'], pm['dets_x'], pm['dets_y'], pm['pixsize'], pm['wavelength'], pm['ewald_rad'], show=args.vb)
    qscaling    = 1. / pm['wavelength'] / q_pm['q_sep']

    # RESCALE CONDOR Qs to match dragonfly convention
    # -----------------------------------------------
    qx = condor_qx / (2 * np.pi) * condor_wavelength * qscaling
    qy = condor_qy / (2 * np.pi) * condor_wavelength * qscaling
    qz = condor_qz / (2 * np.pi) * condor_wavelength * qscaling

    if pm['mask_fname'] is None:
        mask = np.zeros(condor_mask.shape)
        mask [~(condor_mask == 0)] = 2
        rmax = min(condor_X.max(), np.abs(condor_X.min()), condor_Y.max(), np.abs(condor_Y.min()))
        mask[(condor_R>rmax)] = 1
        mask = mask.flatten()
    else:
        mask = np.fromfile(pm['mask_fname'], '=u1')

    det_file = output_folder + '/' + 'condor.dat'
    sys.stderr.write('Writing detector file to %s\n' % det_file)
    
    with open(det_file, "w") as fp:
        fp.write(str(condor_qx.shape[0]) + "\n")
        for t0,t1,t2,t3,t4 in zip(qx,qy,qz,condor_solid_angle,mask):
            txt = "%21.15e %21.15e %21.15e %21.15e %d\n"%(t0, t1, t2, t3, t4)
            fp.write(txt)
