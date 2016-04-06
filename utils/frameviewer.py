#!/usr/bin/env python

import numpy as np
import sys
import os
import ConfigParser
from py_src import py_utils
from py_src import read_config

if __name__ == "__main__":
    # Read detector and photons file from config
    parser = py_utils.my_argparser(description="make detector")
    parser.add_argument('num', help='frame number or filename containing list of frame numbers')
    args = parser.special_parse_args()

    if os.path.isfile(args.num):
        num_list = np.loadtxt(args.num, dtype='i4')
    else:
        num_list = np.array([int(args.num)])
 
    try:
        photons_list = [read_config.get_param(args.config_file, 'emc', "in_photons_file")]
    except ConfigParser.NoOptionError:
        with open(read_config.get_param(args.config_file, 'emc', "in_photons_list"), 'r') as f:
            photons_list = map(lambda x: x.rstrip(), f.readlines())

    pm = read_config.get_detector_config(args.config_file, show=args.vb)

    x, y = np.indices((pm['dets_x'], pm['dets_y']))
    x = x.flatten()
    y = y.flatten()

    num_data_list = []
    ones_accum_list = []
    multi_accum_list = []

    # For each emc file, read num_data and generate ones_accum and multi_accum
    for photons_file in photons_list:
        # Read photon data
        with open(photons_file, 'rb') as f:
            num_data = np.fromfile(f, dtype='i4', count=1)[0]
            f.seek(1024, 0)
            ones = np.fromfile(f, dtype='i4', count=num_data)
            multi = np.fromfile(f, dtype='i4', count=num_data)
        num_data_list.append(num_data)
        ones_accum_list.append(np.add.accumulate(ones))
        multi_accum_list.append(np.add.accumulate(multi))

    num_data_list = np.add.accumulate(num_data_list)
    file_num_list = [np.where(num < num_data_list)[0][0] for num in num_list]
    frame_num_list = []
    for file_num, num in zip(file_num_list, num_list):
        if file_num == 0:
            frame_num_list.append(num)
        else:
            frame_num_list.append(num - num_data_list[file_num-1])
    frames = np.zeros((len(num_list), x.max()+1, y.max()+1), dtype='i4')

    for i, file_num, frame_num in zip(range(len(num_list)), file_num_list, frame_num_list):
        with open(photons_list[file_num], 'rb') as f:
            num_data = np.fromfile(f, dtype='i4', count=1)[0]

            ones_accum = ones_accum_list[file_num]
            multi_accum = multi_accum_list[file_num]
            
            if frame_num == 0:
                ones_offset = 0
                multi_offset = 0
                ones_size = ones_accum[frame_num]
                multi_size = multi_accum[frame_num]
            else:
                ones_offset = ones_accum[frame_num - 1]
                multi_offset = multi_accum[frame_num - 1]
                ones_size = ones_accum[frame_num] - ones_accum[frame_num - 1]
                multi_size = multi_accum[frame_num] - multi_accum[frame_num - 1]

            f.seek(1024 + num_data*8 + ones_offset*4, 0)
            place_ones = np.fromfile(f, dtype='i4', count=ones_size)

            f.seek(1024 + num_data*8 + ones_accum[-1]*4 + multi_offset*4, 0)
            place_multi = np.fromfile(f, dtype='i4', count=multi_size)

            f.seek(1024 + num_data*8 + ones_accum[-1]*4 + multi_accum[-1]*4 + multi_offset*4, 0)
            count_multi = np.fromfile(f, dtype='i4', count=multi_size)

        np.add.at(frames[i], (x[place_ones], y[place_ones]), 1)
        np.add.at(frames[i], (x[place_multi], y[place_multi]), count_multi)
        sys.stderr.write('\rWritten frame %d/%d' % (i+1, len(num_list)))
    sys.stderr.write('\n')

    if len(num_list) > 1:
        prefix = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    else:
        prefix = 'frames'
    print 'Prefix =', prefix
    
    frames.tofile('data/%s_%d_%d.int' % (prefix, frames.shape[1], frames.shape[2]))
    import h5py
    f = h5py.File('data/%s.h5' % prefix, 'w')
    f['data/frames'] = frames
    f['data/file_name'] = [photons_list[i][5:-3]+'h5' for i in file_num_list]
    f['data/frame_num'] = frame_num_list
    f.close()

    if len(num_list) == 1:
        import pylab as P
        P.matshow(frames[0], vmax=25)
        P.show()
