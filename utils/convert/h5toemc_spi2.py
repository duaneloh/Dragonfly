#!/usr/bin/env python

'''
Convert h5 files generated by Chuck for the SPI
These files have also been classified into singles so no selection file is
needed.

Needs:
    <h5_fname> - Path to photon-converted h5 file used in SPI

Produces:
    EMC file with all the single hits in the h5 file
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
    parser      = py_utils.my_argparser(description='h5toemc')
    parser.add_argument('h5_name', help='HDF5 file to convert to emc format')
    parser.add_argument('-d', '--dset_name', help='Name of HDF5 dataset containing photon data', default=None)
    args        = parser.special_parse_args()

    logging.info('Starting h5toemc_spi2....')
    logging.info(' '.join(sys.argv))
    pm          = read_config.get_detector_config(args.config_file, show=args.vb)

    if not os.path.isfile(args.h5_name):
        print 'Data file %s not found. Exiting.' % args.h5_name
        logging.error('Data file %s not found. Exiting.' % args.h5_name)
        sys.exit()

    f = h5py.File(args.h5_name, 'r')
    if args.dset_name is None:
        for name, obj in f['photonConverter'].items():
            try:
                temp = obj.keys()
                dset = obj['photonCount']
                break
            except AttributeError:
                pass
        logging.info('Converting data in '+ dset.name)
    else:
        dset = f[args.dset_name]
        logging.info('Converting data in '+ args.dset_name)
    frames = dset[:]
    ind = range(len(frames))
    num_frames = len(ind)
    logging.info('%d frames in %s' % (num_frames, args.h5_name))

    emcwriter = writeemc.EMC_writer('data/%s.emc' % os.path.splitext(os.path.basename(sys.argv[1]))[0],
                                    pm['dets_x']*pm['dets_y'])

    for i in range(num_frames):
        emcwriter.write_frame(frames[i].flatten().astype('i4'))
        sys.stderr.write('\rFinished %d/%d' % (i+1, num_frames))

    f.close()
    sys.stderr.write('\n')
    emcwriter.finish_write()
