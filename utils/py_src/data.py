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
        
        # For each emc file, read num_data and generate ones_accum and multi_accum
        for photons_file in self.photons_list:
            # Read photon data
            with open(photons_file, 'rb') as f:
                num_data = np.fromfile(f, dtype='i4', count=1)[0]
                f.seek(1024, 0)
                ones = np.fromfile(f, dtype='i4', count=num_data)
                multi = np.fromfile(f, dtype='i4', count=num_data)
            self.num_data_list.append(num_data)
            self.ones_accum_list.append(np.cumsum(ones))
            self.multi_accum_list.append(np.cumsum(multi))
        
        self.num_data_list = np.cumsum(self.num_data_list)
        self.num_frames = self.num_data_list[-1]

    def get_frame(self, num):
        file_num = np.where(num < self.num_data_list)[0][0]
        if file_num == 0:
            frame_num = num
        else:
            frame_num = num - self.num_data_list[file_num-1]
        
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

class Det_reader():
    def __init__(self, det_fname, detd, ewald_rad, mask_flag=False):
        self.det_fname = det_fname
        self.detd = detd
        self.ewald_rad = ewald_rad
        self.init_geom(mask_flag)

    def init_geom(self, mask_flag):
        sys.stderr.write('Reading detector file...')
        if mask_flag:
            sys.stderr.write('with mask...')
            cx, cy, cz, mask = np.loadtxt(self.det_fname, usecols=(0,1,2,4), skiprows=1, unpack=True)
            #mask[mask==2] = 1 # To keep only mask==0
            mask[mask==1] = 0 # To keep both 0 and 1
            mask = mask / 2 # To keep both 0 and 1
            mask = 1 - mask
        else:
            cx, cy, cz = np.loadtxt(self.det_fname, usecols=(0,1,2), skiprows=1, unpack=True)
            mask = np.ones(cx.shape)
        sys.stderr.write('done\n')
        
        cx = cx*self.detd/(cz+self.ewald_rad)
        cy = cy*self.detd/(cz+self.ewald_rad)
        self.x = np.round(cx - cx.min()).astype('i4')
        self.y = np.round(cy - cy.min()).astype('i4')
        
        self.frame_shape = (self.x.max()+1, self.y.max()+1)
        self.mask = np.ones(self.frame_shape)
        self.mask[self.x, self.y] = mask.flatten()
        self.raw_mask = mask
        self.cx = cx
        self.cy = cy
