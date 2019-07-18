#!/usr/bin/env python

'''
Convert PSANA geometry file to detector file

Needs:
    Experiment string (eg. exp=cxim7816:run=10)
    Detector name (eg. DsaCsPad)

Produces:
    Detector file in output_folder
'''

import os
import sys
import logging
import numpy as np
#Add utils directory to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from py_src import py_utils # pylint: disable=wrong-import-position
from py_src import read_config # pylint: disable=wrong-import-position
from py_src import detector # pylint: disable=wrong-import-position

try:
    import psana
except ImportError:
    sys.stderr.write('This utility needs PSANA to run. Unable to import psana module\n')
    sys.exit(1)

def main():
    """Parse command line arguments and generate file"""
    logging.basicConfig(filename='recon.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = py_utils.MyArgparser(description='PSANA to det')
    parser.add_argument('exp_string', help='PSANA experiment string')
    parser.add_argument('det_name', help='PSANA Detector name (either source string or alias)')
    parser.add_argument('-C', '--corners',
                        help='Set corner pixels to be irrelevant. Default = False',
                        action='store_true', default=False)
    args = parser.special_parse_args()

    logging.info('Starting psanatodet...')
    logging.info(' '.join(sys.argv))
    pm = read_config.get_detector_config(args.config_file, show=args.vb) # pylint: disable=invalid-name
    q_pm = read_config.compute_q_params(pm['detd'], pm['dets_x'], pm['dets_y'],
                                        pm['pixsize'], pm['wavelength'],
                                        pm['ewald_rad'], show=args.vb)
    output_folder = read_config.get_filename(args.config_file, 'emc', 'output_folder')

    dsource = psana.DataSource(args.exp_string + ':idx')
    run = dsource.runs().next()
    times = run.times()
    evt = run.event(times[0])
    psana_det = psana.Detector(args.det_name)
    cx = psana_det.coords_x(evt).flatten() * 1.e-3 # pylint: disable=invalid-name
    cy = psana_det.coords_y(evt).flatten() * 1.e-3 # pylint: disable=invalid-name

    detd = pm['detd'] # in mm
    qscaling = 1. / pm['wavelength'] / q_pm['q_sep']
    rad = np.sqrt(cx*cx + cy*cy)
    norm = np.sqrt(cx*cx + cy*cy + detd*detd)
    polar = read_config.compute_polarization(pm['polarization'], cx, cy, norm)

    det = detector.Detector()
    det.detd = detd / pm['pixsize']
    det.ewald_rad = pm['ewald_rad']
    det.qx = cx * qscaling / norm
    det.qy = cy * qscaling / norm
    det.qz = qscaling * (detd / norm - 1.)
    det.corr = pm['detd']*pm['pixsize']*pm['pixsize'] / np.power(norm, 3.0)
    det.corr *= polar

    if pm['mask_fname'] is None:
        det.raw_mask = psana_det.mask(evt, status=True, calib=True, edges=True,
                                      central=True, unbondnbrs8=True).flatten()
        det.raw_mask = 2*(1 - det.raw_mask)
        if args.corners:
            rmax = min(cx.max(), np.abs(cx.min()), cy.max(), np.abs(cy.min()))
            det.raw_mask[(rad > rmax) & (det.raw_mask == 0)] = 1
    else:
        det.raw_mask = np.fromfile(pm['mask_fname'], '=u1')

    det_file = output_folder + '/det_' + args.exp_string.split(':')[0][4:]
    try:
        import h5py
        det_file += '.h5'
    except ImportError:
        det_file += '.dat'
    sys.stderr.write('Writing detector file to %s\n' % det_file)
    det.write(det_file)

if __name__ == '__main__':
    main()
