import sys
import os
import numpy as np

class EMC_reader():
    def __init__(self, photons_list, x, y, mask):
        self.photons_list = photons_list
        self.x = x
        self.y = y
        self.mask = mask
        self.frame_shape = mask.shape
        self.parse_headers()

    def parse_headers(self):
        self.num_data_list = []
        self.ones_accum_list = []
        self.multi_accum_list = []
        self.num_pix = []
        
        for photons_file in self.photons_list:
            with open(photons_file, 'rb') as f:
                num_data = np.fromfile(f, dtype='i4', count=1)[0]
                self.num_pix.append(np.fromfile(f, dtype='i4', count=1)[0])
                if self.num_pix[-1] != self.num_pix[0]:
                    sys.stderr.write('Warning: num_pix for %s is different (%d vs %d)' % (photons_file, self.num_pix[-1], self.num_pix[0]))
                f.seek(1024, 0)
                ones = np.fromfile(f, dtype='i4', count=num_data)
                multi = np.fromfile(f, dtype='i4', count=num_data)
            self.num_data_list.append(num_data)
            self.ones_accum_list.append(np.cumsum(ones))
            self.multi_accum_list.append(np.cumsum(multi))
        
        self.num_data_list = np.cumsum(self.num_data_list)
        self.num_frames = self.num_data_list[-1]

    def get_frame(self, num, raw=True):
        file_num = np.where(num < self.num_data_list)[0][0]
        if file_num == 0:
            frame_num = num
        else:
            frame_num = num - self.num_data_list[file_num-1]
        
        if raw:
            return self.read_frame(file_num, frame_num)
        else:
            return self.read_frame(file_num, frame_num)

    def read_frame(self, file_num, frame_num):
        with open(self.photons_list[file_num], 'rb') as f:
            num_data = np.fromfile(f, dtype='i4', count=1)[0]
            
            ones_accum = self.ones_accum_list[file_num]
            multi_accum = self.multi_accum_list[file_num]
            
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
        
        frame = np.zeros(self.frame_shape, dtype='i4')
        np.add.at(frame, (self.x[place_ones], self.y[place_ones]), 1)
        np.add.at(frame, (self.x[place_multi], self.y[place_multi]), count_multi)
        
        return frame * self.mask

    def read_raw_frame(self, file_num, frame_num):
        with open(self.photons_list[file_num], 'rb') as f:
            num_data = np.fromfile(f, dtype='i4', count=1)[0]
            
            ones_accum = self.ones_accum_list[file_num]
            multi_accum = self.multi_accum_list[file_num]
            
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
        
        frame = np.zeros(self.num_pix[file_num], dtype='i4')
        frame[self.x[place_ones], self.y[place_ones]] = 1
        frame[self.x[place_multi], self.y[place_multi]] = 1

