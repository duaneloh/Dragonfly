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
    if args.dset_name is None:
        for name, obj in fp['photonConverter'].items():
            try:
                temp = obj.keys()
                dset = obj['photonCount']
                break
            except AttributeError:
                pass
        logging.info('Converting data in '+ dset.name)
    else:
        try:
            dset = fp[args.dset_name]
        except KeyError:
            print 'Dataset not found. Moving on.'
            return None 
        logging.info('Converting data in '+ args.dset_name)
    return dset

def get_indices(fp, dset, args):
    all_frames = False
    if args.sel_file is not None and args.sel_dset is not None:
        logging.info('Both sel_file and sel_dset specified. Pick one.')
        sys.exit(1)
    elif args.sel_file is None and args.sel_dset is None:
        logging.info('Converting all images. dset.shape = %s' % (dset.shape,))
        if len(dset.shape) == 3:
            ind = np.arange(dset.shape[0], dtype='i4')
        else:
            ind = 0
        all_frames = True
    elif args.sel_file is not None:
        ind = np.loadtxt(args.sel_file, dtype='i4')
        ind -= curr_num_data
    else:
        ind = fp[args.sel_dset][:]

    if type(ind) is np.ndarray:
        if not all_frames and ind.shape[0] == dset.shape[0] and ind.max() < 2:
            ind = np.where(ind==1)[0]
        ind = ind[(ind>=0) & (ind<dset.shape[0])]
        num_frames = ind.shape[0]
    else:
        num_frames = 1

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
    parser.add_argument('-d', '--dset_name', help='Name of HDF5 dataset containing photon data', default=None)
    parser.add_argument('-s', '--sel_file', help='Path to text file containing indices of frames\nor a set of 0 or 1 values. Default: Do all', default=None)
    parser.add_argument('-S', '--sel_dset', help='Same as --sel_file, but pointing to the name of an HDF5 dataset', default=None)
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
        emcwriter = writeemc.EMCWriter('%s/%s.emc' % (output_folder, os.path.splitext(os.path.basename(args.h5_name))[0]),
                                        pm['dets_x']*pm['dets_y'])
    else:
        emcwriter = writeemc.EMCWriter(args.out_fname, pm['dets_x']*pm['dets_y'])

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
            if len(dset.shape) == 3:
                photons = dset[ind[i]]
            else:
                photons = dset[:]
            photons[photons<0] = 0
            emcwriter.write_frame(photons.flatten())
            if not args.list:
                sys.stderr.write('\rFinished %d/%d' % (i+1, num_frames))

        f.close()
        if not args.list:
            sys.stderr.write('\n')
        sys.stderr.write('\rProcessed %s %d/%d' % (fname, fnum, len(flist)))

    sys.stderr.write('\n')
    emcwriter.finish_write()
