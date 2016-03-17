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
sys.path.append('../utils')
from py_src import writeemc

if len(sys.argv) < 2:
    print "Format: %s <h5_fname>" % sys.argv[0]
    sys.exit(1)

if not os.path.isfile(sys.argv[1]):
    print 'Data file %s not found. Exiting.' % sys.argv[1]
    print
    sys.exit()

f = h5py.File(sys.argv[1], 'r')
frames = f['photonConverter/pnccdBack/photonCount'][:]
leastsq = f['particleCorrelator/leastSq'][:]
std = f['angularAverage/std'][:]
ind = np.where((leastsq < 20.) & (std < 0.35))[0]
num_frames = len(ind)
print num_frames, "frames in h5 file"

emcwriter = writeemc.EMC_writer(['data/temp.po', 'data/temp.pm', 'data/temp.cm'], 
                                'data/%s.emc' % os.path.splitext(os.path.basename(sys.argv[1]))[0],
                                257*260)

for i in range(num_frames):
    emcwriter.write_frame(frames[ind[i]].T.flatten().astype('i4'))
    sys.stderr.write('\rFinished %d/%d' % (i+1, num_frames))

f.close()
sys.stderr.write('\n')
emcwriter.finish_write()
