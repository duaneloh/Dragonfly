#!/usr/bin/env python

'''
Convert Cheetah geometry file to detector file
Can specify mask file separately.

Needs:
    <h5_fname> - Path to Cheetah geometry h5 file

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

if __name__ == '__main__':
    logging.basicConfig(filename='recon.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = py_utils.my_argparser(description='cheetahtodet')
    parser.add_argument('h5_name', help='HDF5 file to convert to detector format')
    parser.add_argument('-M', '--mask', help='Path to detector style mask (0:good, 1:no_orient, 2:bad) in h5 file')
    parser.add_argument('--mask_dset', help='Data set in mask file. Default: /data/data', default='data/data')
    args = parser.special_parse_args()

    logging.info('Starting cheetahtodet...')
    logging.info(' '.join(sys.argv))
    pm = read_config.get_detector_config(args.config_file, show=args.vb)
    q_pm = read_config.compute_q_params(pm['detd'], pm['dets_x'], pm['dets_y'], pm['pixsize'], pm['wavelength'], pm['ewald_rad'], show=args.vb)
    output_folder = read_config.get_filename(args.config_file, 'emc', 'output_folder')

    # Cheetah geometry files have coordinates in m
    with h5py.File(args.h5_name, 'r') as f:
        x = f['x'][:].flatten() * 1.e3
        y = f['y'][:].flatten() * 1.e3
        z = f['z'][:].flatten() * 1.e3 + pm['detd']
    
    norm = np.sqrt(x*x + y*y + z*z)
    polar = read_config.compute_polarization(pm['polarization'], x, y, norm)
    qscaling    = 1. / pm['wavelength'] / q_pm['q_sep']
    qx = x * qscaling / norm
    qy = y * qscaling / norm
    qz = qscaling * (z / norm - 1.)
    solid_angle = pm['detd']*(pm['pixsize']*pm['pixsize']) / np.power(norm, 3.0)
    solid_angle = polar*solid_angle
    radius = np.sqrt(x*x + y*y)
    if args.mask is None:
        rmax = min(np.abs(x.max()), np.abs(x.min()), np.abs(y.max()), np.abs(y.min()))
        mask = np.zeros(solid_angle.shape, dtype='u1')
        mask[radius>rmax] = 1
    else:
        with h5py.File(args.mask, 'r') as f:
            mask = f[args.mask_dset][:].astype('u1').flatten()
    
    det_file = output_folder + '/' + os.path.splitext(os.path.basename(args.h5_name))[0] + '.dat'
    logging.info('Writing detector file to %s'%det_file)
    sys.stderr.write('Writing detector file to %s\n'%det_file)
    
    with open(det_file, "w") as fp:
        fp.write(str(qx.shape[0]) + "\n")
        for t0,t1,t2,t3,t4 in zip(qx,qy,qz,solid_angle,mask):
            txt = "%21.15e %21.15e %21.15e %21.15e %d\n"%(t0, t1, t2, t3, t4)
            fp.write(txt)
