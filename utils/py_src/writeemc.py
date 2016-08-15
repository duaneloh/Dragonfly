import numpy as np
import struct
import os

class EMC_writer():
    def __init__(self, out_fname, num_pix):
        out_folder = os.path.dirname(out_fname)
        temp_fnames = [os.path.join(out_folder, fname) for fname in ['temp.po', 'temp.pm', 'temp.cm']]
        self.f = [open(fname, 'wb') for fname in temp_fnames]
        
        self.out_fname = out_fname
        print 'Writing emc file to', out_fname
        self.num_data = 0
        self.num_pix = num_pix
        self.mean_count = 0.
        self.ones = []
        self.multi = []

    def finish_write(self):
        for fp in self.f:
            fp.close()
        
        if self.num_data == 0:
            for fp in self.f:
                os.system('rm ' + fp.name)
            return
        
        self.mean_count /= self.num_data
        print 'num_data = %d, mean_count = %.4e' % (self.num_data, self.mean_count)
        self.ones = np.array(self.ones)
        self.multi = np.array(self.multi)
        
        fp = open(self.out_fname, 'wb')
        header = np.zeros((256), dtype='i4')
        header[0] = self.num_data
        header[1] = self.num_pix
        header.tofile(fp)
        self.ones.astype('i4').tofile(fp)
        self.multi.astype('i4').tofile(fp)
        fp.close()
        for fp in self.f:
            os.system('cat ' + fp.name + ' >> ' + self.out_fname)
            os.system('rm ' + fp.name)

    def write_frame(self, frame):
        place_ones = np.where(frame == 1)[0]
        place_multi = np.where(frame > 1)[0]
        count_multi = frame[place_multi]
        
        self.num_data += 1
        self.mean_count += len(place_ones) + count_multi.sum()
        self.ones.append(len(place_ones))
        self.multi.append(len(place_multi))
        
        #self.f[0].write(place_ones.astype(np.int32).tostring())
        #self.f[1].write(place_multi.astype(np.int32).tostring())
        #self.f[2].write(count_multi.astype(np.int32).tostring())
        place_ones.astype(np.int32).tofile(self.f[0])
        place_multi.astype(np.int32).tofile(self.f[1])
        count_multi.astype(np.int32).tofile(self.f[2])

