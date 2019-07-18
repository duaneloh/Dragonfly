#!/usr/bin/env python

'''
Convert CrystFEL geometry file to detector file
Can specify mask file separately.

Needs:
    <geom_fname> - Path to CrystFEL geometry h5 file

Produces:
    Detector file in output_folder
'''

from __future__ import print_function
import sys
import os
import logging
import numpy as np
import h5py
#Add utils directory to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from py_src import py_utils # pylint: disable=wrong-import-position
from py_src import read_config # pylint: disable=wrong-import-position
from py_src import detector # pylint: disable=wrong-import-position
try:
    from cfelpyutils import crystfel_utils, geometry_utils
except ImportError:
    print('Need cfelpyutils package to safely parse geometry file.')
    print('Install from pip if possible.')
    raise

def main():
    """Parse command line arguments and convert file"""
    logging.basicConfig(filename='recon.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = py_utils.MyArgparser(description='cheetahtodet')
    parser.add_argument('geom_fname', help='CrystFEL geometry file to convert to detector format')
    parser.add_argument('-M', '--mask',
                        help='Path to detector style mask (0:good, 1:no_orient, 2:bad) in h5 file')
    parser.add_argument('--mask_dset',
                        help='Data set in mask file. Default: /data/data', default='data/data')
    parser.add_argument('--dragonfly_mask',
                        help='Whether mask has Dragonfly style values or not. (Default: false)',
                        default=False, action='store_true')
    args = parser.special_parse_args()

    logging.info('Starting cheetahtodet...')
    logging.info(' '.join(sys.argv))
    pm = read_config.get_detector_config(args.config_file, show=args.vb) # pylint: disable=invalid-name
    q_pm = read_config.compute_q_params(pm['detd'], pm['dets_x'], pm['dets_y'],
                                        pm['pixsize'], pm['wavelength'],
                                        pm['ewald_rad'], show=args.vb)
    output_folder = read_config.get_filename(args.config_file, 'emc', 'output_folder')

    # CrystFEL geometry files have coordinates in pixel size units
    geom = crystfel_utils.load_crystfel_geometry(args.geom_fname)
    pixmap = geometry_utils.compute_pix_maps(geom)
    x = pixmap.x
    y = pixmap.y
    z = pm['detd'] / pm['pixsize']
    pm['pixsize'] = 1.
    
    det = detector.Detector()
    norm = np.sqrt(x*x + y*y + z*z)
    qscaling = 1. / pm['wavelength'] / q_pm['q_sep']
    det.qx = x * qscaling / norm
    det.qy = y * qscaling / norm
    det.qz = qscaling * (z / norm - 1.)
    det.corr = pm['detd']*(pm['pixsize']*pm['pixsize']) / np.power(norm, 3.0)
    det.corr *= read_config.compute_polarization(pm['polarization'], x, y, norm)
    if args.mask is None:
        radius = np.sqrt(x*x + y*y)
        rmax = min(np.abs(x.max()), np.abs(x.min()), np.abs(y.max()), np.abs(y.min()))
        det.raw_mask = np.zeros(det.corr.shape, dtype='u1')
        det.raw_mask[radius > rmax] = 1
    else:
        with h5py.File(args.mask, 'r') as fptr:
            det.raw_mask = fptr[args.mask_dset][:].astype('u1').flatten()
        if not args.dragonfly_mask:
            det.raw_mask = 2 - 2*det.raw_mask

    det.detd = pm['detd'] / pm['pixsize']
    det.ewald_rad = pm['ewald_rad']
    det_file = output_folder + '/' + os.path.splitext(os.path.basename(args.geom_fname))[0]
    try:
        import h5py
        det_file += '.h5'
    except ImportError:
        det_file += '.dat'
    logging.info('Writing detector file to %s', det_file)
    sys.stderr.write('Writing detector file to %s\n'%det_file)
    det.write(det_file)

if __name__ == '__main__':
    main()
