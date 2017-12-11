#!/usr/bin/env python

'''
Convert frames in an h5 file or list of files into emc format
If not all frames need to be converted, one can use a selection file or a 
selection dataset

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

def get_dset(fp, args):
    try:
        dset = fp['hits/unassembled']
    except KeyError:
        print 'Dataset not found. Moving on.'
        return None 
    return dset

def get_indices(fp, dset, args):
    all_frames = False
    ind = np.where(fp['hits/litPixels'][:] > args.thresh)[0]

    if type(ind) is np.ndarray:
        if not all_frames and ind.shape[0] == dset.shape[0] and ind.max() < 2:
            ind = np.where(ind==1)[0]
        ind = ind[(ind>=0) & (ind<dset.shape[0])]
        num_frames = ind.shape[0]
    else:
        num_frames = 1

    print 'Converting %d frames' % num_frames
    return ind, num_frames

def bin_image(array, binning):
    '''
    Convenience function to bin 2D array by some bin factor
    The binning must divide the array shape
    '''
    s = array.shape
    out = array.reshape(s[0]/binning, binning, s[1]/binning, binning).sum(axis=(1,3))
    return out

if __name__ == '__main__':
    logging.basicConfig(filename='recon.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = py_utils.my_argparser(description='h5toemc')
    parser.add_argument('h5_name', help='HDF5 file to convert to emc format')
    parser.add_argument('-t', '--thresh', help='Lit pixel threshold for good hits', default=0, type=int)
    parser.add_argument('-l', '--list', help='h5_name is list of h5 files rather than a single one', action='store_true', default=False)
    parser.add_argument('-o', '--out_fname', help='Output filename if different from calculated name', default=None)
    args = parser.special_parse_args()

    logging.info('Starting h5toemc....')
    logging.info(' '.join(sys.argv))
    pm = read_config.get_detector_config(args.config_file, show=args.vb)
    output_folder = read_config.get_filename(args.config_file, 'emc', 'output_folder')
    curr_num_data = 0

    if not os.path.isfile(args.h5_name):
        print 'Data file %s not found. Exiting.' % args.h5_name
        logging.error('Data file %s not found. Exiting.' % args.h5_name)
        sys.exit()

    if args.list:
        logging.info('Reading file names in list %s' % args.h5_name)
        with open(args.h5_name, 'r') as f:
            flist = [os.path.realpath(fname.rstrip()) for fname in f.readlines()]
        logging.info
    else:
        flist = [args.h5_name]

    if args.out_fname is None:
        emcwriter = writeemc.EMC_writer('%s/%s.emc' % (output_folder, os.path.splitext(os.path.basename(args.h5_name))[0]),
                                        pm['dets_x']*pm['dets_y'])
    else:
        emcwriter = writeemc.EMC_writer(args.out_fname, pm['dets_x']*pm['dets_y'])

    for fnum, fname in enumerate(flist):
        f = h5py.File(fname, 'r')
        dset = get_dset(f, args)
        if dset is None:
            continue
        
        ind, num_frames = get_indices(f, dset, args)

        curr_num_data += num_frames
        if not args.list:
            logging.info('Converting %d/%d frames in %s' % (num_frames, dset.shape[0], args.h5_name))

        for i in range(num_frames):
            data = dset[ind[i],:,:,:]
            photons = np.concatenate((data[15,256:], data[3,384:], data[4,384:]))
            photons = np.floor(photons/65. + 0.4).astype('i4')
            photons[photons<0] = 0
            if photons.sum() == 0:
                continue
            emcwriter.write_frame(photons.flatten())
            if not args.list:
                sys.stderr.write('\rFinished %d/%d' % (i+1, num_frames))

        f.close()
        if not args.list:
            sys.stderr.write('\n')
        sys.stderr.write('\rProcessed %s %d/%d' % (fname, fnum, len(flist)))

    sys.stderr.write('\n')
    emcwriter.finish_write()
