#!/usr/bin/env python

import numpy as np
import h5py
import sys
from py_src import writeemc

if len(sys.argv) < 2:
    print "Format: %s <file_list>" % sys.argv[0]
    sys.exit(1)

flist = np.loadtxt(sys.argv[1], dtype='S')
print len(flist)
emcwriter = writeemc.EMC_writer(['temp.po', 'temp.pm', 'temp.cm'], 'photons.emc', 1700**2)

i = 1
for fname in flist:
    f = h5py.File(fname, 'r')
    pattern = np.round(f['data/data'][:]).astype(np.int32).flatten()
    #pattern[pattern > 1] = 1
    f.close()
    
    emcwriter.write_frame(pattern)
    sys.stderr.write('\rFinished %d/%d' % (i, len(flist)))
    i += 1

emcwriter.finish_write()
