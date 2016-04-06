#!/usr/bin/env python

'''
Convert h5 files generated by pattern_sim to EMC format

Needs:
    <file_list> - Text file with one h5 file per line

Produces:
    photons.emc - EMC file with all the file in the file list
'''

import os
import numpy as np
import h5py
import sys
sys.path.append('../utils')
from py_src import writeemc

if len(sys.argv) < 2:
    print "Format: %s <file_list>" % sys.argv[0]
    print "\tOptional: binning"
    sys.exit(1)

binsize = 1
def binimg(array, bs):
	s = array.shape
	out = array.reshape(s[0]/bs, bs, s[1]/bs, bs).sum(axis=(1,3))
	return out

tag = ''
if len(sys.argv) > 2:
    binsize = int(sys.argv[2])
    tag = '_%d' % binsize

num_pix = (1700/binsize)**2
print num_pix, 'pixels in detector'

flist = np.loadtxt(sys.argv[1], dtype='S')
print len(flist), 'files in list'
emcwriter = writeemc.EMC_writer(['temp.po', 'temp.pm', 'temp.cm'], 
                                'data/%s%s.emc' % (os.path.splitext(os.path.basename(sys.argv[1]))[0], tag), 
                                num_pix)

i = 1
for fname in flist:
    f = h5py.File(fname, 'r')
    pattern = (f['data/data'][:]*1.e-8).astype('i4')
    #pattern = f['data/data'][:].astype('i4')
    pattern = binimg(pattern, binsize)
    #pattern[pattern > 1] = 1
    f.close()
    
    emcwriter.write_frame(pattern.flatten())
    sys.stderr.write('\rFinished %d/%d' % (i, len(flist)))
    i += 1

emcwriter.finish_write()
