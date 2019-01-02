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
    import psana
except ImportError:
    sys.stderr.write('This utility needs PSANA to run. Unable to import psana module\n')
    sys.exit(1)

if __name__ == '__main__':
    logging.basicConfig(filename='recon.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = py_utils.my_argparser(description='PSANA to det')
    parser.add_argument('exp_string', help='PSANA experiment string')
    parser.add_argument('det_name', help='Detector name (either source string or alias)')
    args = parser.special_parse_args()

    logging.info('Starting psanatodet...')
    logging.info(' '.join(sys.argv))
    pm = read_config.get_detector_config(args.config_file, show=args.vb)
    q_pm = read_config.compute_q_params(pm['detd'], pm['dets_x'], pm['dets_y'], pm['pixsize'], pm['wavelength'], pm['ewald_rad'], show=args.vb)
    output_folder = read_config.get_filename(args.config_file, 'emc', 'output_folder')

    ds = psana.DataSource(args.exp_string + ':idx')
    run = ds.runs().next()
    times = run.times()
    evt = run.event(times[0])
    det = psana.Detector(args.det_name)
    cx = det.coords_x(evt).flatten() * 1.e-3
    cy = det.coords_y(evt).flatten() * 1.e-3

    detd = pm['detd']
    qscaling = 1. / pm['wavelength'] / q_pm['q_sep']
    rad = np.sqrt(cx*cx + cy*cy)
    norm = np.sqrt(cx*cx + cy*cy + detd*detd)
    polar = read_config.compute_polarization(pm['polarization'], cx, cy, norm)
    qx = cx * qscaling / norm
    qy = cy * qscaling / norm
    qz = qscaling * (detd / norm - 1.)
    solid_angle = pm['detd']*pm['pixsize']*pm['pixsize'] / np.power(norm, 3.0)
    solid_angle = polar*solid_angle

    if pm['mask_fname'] is None:
        mask = det.mask(evt, status=True, calib=True, edges=True, central=True, unbondnbrs8=True).flatten()
        mask = 2*(1 - mask)
        rmax = min(cx.max(), np.abs(cx.min()), cy.max(), np.abs(cy.min()))
        mask[(rad>rmax) & (mask==0)] = 1
    else:
        mask = np.fromfile(pm['mask_fname'], '=u1')

    det_file = output_folder + '/det_' + args.exp_string.split(':')[0][4:] + '.dat'
    sys.stderr.write('Writing detector file to %s\n' % det_file)
    
    with open(det_file, "w") as fp:
        fp.write("%d %f %f\n" % (qx.shape[0], detd/pm['pixsize'], qscaling))
        for t0,t1,t2,t3,t4 in zip(qx,qy,qz,solid_angle,mask):
            txt = "%21.15e %21.15e %21.15e %21.15e %d\n"%(t0, t1, t2, t3, t4)
            fp.write(txt)
