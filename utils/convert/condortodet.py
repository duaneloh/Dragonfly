#!/usr/bin/env python

'''
Convert CONDOR geometry conventions to detector file

Needs:
    CXI file produced by CONDOR (eg. photons.cxi)

Produces:
    Detector file in output_folder
'''

import os
import sys
import logging
import numpy as np
import h5py
#Add utils directory to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from py_src import py_utils # pylint: disable=wrong-import-position
from py_src import read_config # pylint: disable=wrong-import-position
from py_src import detector # pylint: disable=wrong-import-position

try:
    import condor
except ImportError:
    sys.stderr.write('This utility needs CONDOR to run. Unable to import condor module\n')
    sys.exit(1)

def _test_match(cval, dval):
    assert cval == dval, "Missmatch between condor file (%f) and config.ini (%f)" %(cval, dval)

def main():
    """Parse command line arguments and convert file"""
    logging.basicConfig(filename='recon.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = py_utils.MyArgparser(description='CONDOR to det')
    parser.add_argument('filename', help='CONDOR file', type=str)
    parser.add_argument('-C', '--corners',
                        help='Set corner pixels to be irrelevant. Default = False',
                        action='store_true', default=False)
    args = parser.special_parse_args()

    logging.info('Starting condortodet...')
    logging.info(' '.join(sys.argv))
    pm = read_config.get_detector_config(args.config_file, show=args.vb) # pylint: disable=invalid-name
    output_folder = read_config.get_filename(args.config_file, 'emc', 'output_folder')
    det = detector.Detector()

    with h5py.File(args.filename, 'r') as fptr:
        condor_dets_x = fptr['detector']['nx'][0]
        condor_dets_y = fptr['detector']['ny'][0]
        condor_detc_x = fptr['detector']['cx'][0]
        condor_detc_y = fptr['detector']['cy'][0]
        condor_detd = fptr['detector']['distance'][0] # in units of m
        condor_pixsize = fptr['detector']['pixel_size'][0] # in units of m
        condor_wavelength = fptr['source/wavelength'][0] # in units of m
        condor_mask = fptr['entry_1/data_1/mask'][0]
    _test_match(condor_dets_x, pm['dets_x'])
    _test_match(condor_dets_y, pm['dets_y'])
    _test_match(condor_detc_x, pm['detc_x'])
    _test_match(condor_detc_y, pm['detc_y'])
    _test_match(condor_detd * 1e3, pm['detd'])
    _test_match(condor_pixsize * 1e3, pm['pixsize'])
    _test_match(condor_wavelength * 1e10, pm['wavelength'])

    # CONDOR CALCULATION OF QMAP AND SOLID ANGLE
    # ---------------------------------------------
    cdet = condor.Detector(condor_detd, condor_pixsize,
                           cx=condor_detc_x, cy=condor_detc_y,
                           nx=condor_dets_x, ny=condor_dets_y)
    condor_X, condor_Y = cdet.generate_xypix(cx=cdet.get_cx_mean_value(), # pylint: disable=invalid-name
                                             cy=cdet.get_cy_mean_value())
    condor_R = np.sqrt(condor_X**2 + condor_Y**2) # pylint: disable=invalid-name
    condor_qmap = cdet.generate_qmap(condor_wavelength, cx=cdet.get_cx_mean_value(),
                                     cy=cdet.get_cy_mean_value(), order='xyz')
    condor_qx = condor_qmap[:, :, 0].flatten()
    condor_qy = condor_qmap[:, :, 1].flatten()
    condor_qz = condor_qmap[:, :, 2].flatten()
    condor_polar = cdet.calculate_polarization_factors(cx=cdet.get_cx_mean_value(),
                                                       cy=cdet.get_cy_mean_value(),
                                                       polarization='unpolarized').flatten()
    det.corr = condor_polar * cdet.get_all_pixel_solid_angles(cx=cdet.get_cx_mean_value(),
                                                              cy=cdet.get_cy_mean_value()).flatten()

    # DRAGONFLY CALCULATION OF QMAP AND SOLID ANGLE
    # ---------------------------------------------
    q_pm = read_config.compute_q_params(pm['detd'], pm['dets_x'], pm['dets_y'],
                                        pm['pixsize'], pm['wavelength'],
                                        pm['ewald_rad'], show=args.vb)
    qscaling = 1. / pm['wavelength'] / q_pm['q_sep']

    # RESCALE CONDOR Qs to match dragonfly convention
    # -----------------------------------------------
    det.qx = condor_qx / (2 * np.pi) * condor_wavelength * qscaling
    det.qy = condor_qy / (2 * np.pi) * condor_wavelength * qscaling
    det.qz = condor_qz / (2 * np.pi) * condor_wavelength * qscaling

    if pm['mask_fname'] is None:
        det.raw_mask = np.zeros(condor_mask.shape)
        det.raw_mask[~(condor_mask == 0)] = 2
        if args.corners:
            rmax = min(condor_X.max(), np.abs(condor_X.min()),
                       condor_Y.max(), np.abs(condor_Y.min()))
            det.raw_mask[condor_R > rmax] = 1
        det.raw_mask = det.raw_mask.flatten()
    else:
        det.raw_mask = np.fromfile(pm['mask_fname'], '=u1')

    det.detd = pm['detd'] / pm['pixsize']
    det.ewald_rad = pm['ewald_rad']
    det_file = output_folder + '/' + 'condor.dat'
    sys.stderr.write('Writing detector file to %s\n' % det_file)
    det.write(det_file)

if __name__ == '__main__':
    main()
