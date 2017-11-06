import sys
import os
import numpy as np

class EMC_reader():
    """EMC file reader class
    Provides access to assembled or raw frames given a list of emc filenames
    
    __init__ arguments:
        photons_list (list of strings) - List of paths to emc files. If single
                                         file, pass as [fname]
        geom_list (list of strings) - List of Det_reader objects.
        geom_mapping (list, optional) - Mapping from photons_list to geom_list
    If there is only one entry in geom_list, all emc files are assumed to point
    to that detector. Otherwise, a mapping must be provided.
    The mapping is a list of the same length as photons_list with entries 
    giving indices in geom_list for the corresponding emc file.
    
    Methods:
        get_frame(num, raw=False)
        get_powder(raw=False)
    """
    def __init__(self, photons_list, geom_list, geom_mapping=None):
        self.photons_list = photons_list
        self.multiple_geom = False
        if len(geom_list) == 1:
            self.geom_list = geom_list * len(photons_list)
        else:
            try:
                self.geom_list = [geom_list[i] for i in geom_mapping]
                self.multiple_geom = True
            except TypeError:
                print('Need mapping if multiple geometries are provided')
                raise
        
        self._parse_headers()

    def get_frame(self, num, raw=False):
        """Get particular frame from file list
        The method determines the file with that frame number and reads it
        
        Arguments:
            num (int) - Frame number 
            raw (bool, optional) - Whether to get unassembled frame
        
        Returns:
            Assembled or unassembled frame as a dense array
        """
        file_num = np.where(num < self.num_data_list)[0][0]
        if file_num == 0:
            frame_num = num
        else:
            frame_num = num - self.num_data_list[file_num-1]
        
        return self._read_frame(file_num, frame_num, raw_flag=raw)

    def get_powder(self, raw=False):
        """Get virtual powder sum of all frames in file list
        
        Arguments:
            raw (bool, optional) - Whether to return unassembled powder sum
        
        Returns:
            Assembled or unassembled powder sum as a dense array
        """
        if self.multiple_geom:
            raise ValueError('Powder sum unreasonable with multiple geometries')
        powder = np.zeros((self.num_pix[0],), dtype='f8')
        
        for photons_file in self.photons_list:
            with open(photons_file, 'rb') as f:
                num_data = np.fromfile(f, dtype='i4', count=1)[0]
                f.seek(1024, 0)
                ones = np.fromfile(f, dtype='i4', count=num_data)
                multi = np.fromfile(f, dtype='i4', count=num_data)
                place_ones = np.fromfile(f, dtype='i4', count=ones.sum())
                place_multi = np.fromfile(f, dtype='i4', count=multi.sum())
                count_multi = np.fromfile(f, dtype='i4', count=multi.sum())
        
            np.add.at(powder, place_ones, 1)
            np.add.at(powder, place_multi, count_multi)
        
        powder *= self.geom_list[0].unassembled_mask
        if not raw:
            powder = self._assemble_frame(powder, self.geom_list[0])
        return powder

    def _parse_headers(self):
        self.num_data_list = []
        self.ones_accum_list = []
        self.multi_accum_list = []
        self.num_pix = []
        
        for i, photons_file in enumerate(self.photons_list):
            with open(photons_file, 'rb') as f:
                num_data = np.fromfile(f, dtype='i4', count=1)[0]
                self.num_pix.append(np.fromfile(f, dtype='i4', count=1)[0])
                if self.num_pix[i] != len(self.geom_list[i].x):
                    sys.stderr.write('Warning: num_pix for %s is different (%d vs %d)\n' % (photons_file, self.num_pix[i], len(self.geom_list[i].x)))
                f.seek(1024, 0)
                ones = np.fromfile(f, dtype='i4', count=num_data)
                multi = np.fromfile(f, dtype='i4', count=num_data)
            self.num_data_list.append(num_data)
            self.ones_accum_list.append(np.cumsum(ones))
            self.multi_accum_list.append(np.cumsum(multi))
        
        self.num_data_list = np.cumsum(self.num_data_list)
        self.num_frames = self.num_data_list[-1]

    def _read_frame(self, file_num, frame_num, raw_flag):
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
        np.add.at(frame, place_ones, 1)
        np.add.at(frame, place_multi, count_multi)
        frame *= self.geom_list[file_num].unassembled_mask
        if not raw_flag:
            frame = self._assemble_frame(frame, self.geom_list[file_num])
        return frame

    def _assemble_frame(self, data, geom):
        img = np.zeros(geom.frame_shape)
        np.add.at(img, [geom.x, geom.y], data)
        return img

