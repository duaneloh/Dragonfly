'''Module containing EMCReader class to parse .emc files'''

from __future__ import print_function
import sys
import numpy as np
import numpy.ma as ma
try:
    import h5py
    HDF5_MODE = True
except ImportError:
    HDF5_MODE = False

class EMCReader(object):
    """EMC file reader
    Provides access to assembled or raw frames given a list of .emc filenames

    __init__ arguments:
        photons_list (list of strings) - List of paths to emc files. If single
                                         file, pass as [fname]
        geom_list (list of strings) - List of DetReader objects.
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
        self.flist = [{'fname': fname} for fname in photons_list]
        num_files = len(photons_list)
        self.multiple_geom = False
        if len(geom_list) == 1:
            for i in range(num_files):
                self.flist[i]['geom'] = geom_list[0]
        else:
            try:
                for i in range(num_files):
                    self.flist[i]['geom'] = geom_list[geom_mapping[i]]
                self.multiple_geom = True
            except TypeError:
                print('Need mapping if multiple geometries are provided')
                raise

        self._assembled_masks = [self._assemble_frame(p['geom'].unassembled_mask, n, fresh=True)
                                 for n, p in enumerate(self.flist)]
        self._parse_headers()

    def _test_h5file(self, fname):
        if HDF5_MODE:
            return h5py.is_hdf5(fname)
        elif os.path.splitext(det_fname)[1] == '.h5':
            fheader = np.fromfile(det_fname, '=c', count=8)
            if fheader == chr(137)+'HDF\r\n'+chr(26)+'\n':
                return True
            else:
                return False
        else:
            return False

    def _parse_headers(self):
        for i, pdict in enumerate(self.flist):
            geom = pdict['geom']
            xsel, ysel = geom.x[geom.unassembled_mask.astype(np.bool)], geom.y[geom.unassembled_mask.astype(np.bool)]
            pdict['zoom_bounds'] = (xsel.min(), xsel.max()+1, ysel.min(), ysel.max()+1)
            
            pdict['is_hdf5'] = self._test_h5file(pdict['fname'])
            if pdict['is_hdf5'] and not HDF5_MODE:
                print('Unable to parse HDF5 dataset')
                raise(IOError)
            elif not pdict['is_hdf5']:
                self._parse_binaryheader(pdict)
            else:
                self._parse_h5header(pdict)

            if pdict['num_pix'] != len(pdict['geom'].x):
                sys.stderr.write(
                    'Warning: num_pix for %s is different (%d vs %d)\n' %
                    (pdict['fname'], pdict['num_pix'], len(pdict['geom'].x)))
            if i > 0:
                pdict['num_data'] += self.flist[i-1]['num_data']
        self.num_frames = self.flist[-1]['num_data']

    def _parse_binaryheader(self, pdict):
        with open(pdict['fname'], 'rb') as fptr:
            num_data = np.fromfile(fptr, dtype='i4', count=1)[0]
            pdict['num_pix'] = np.fromfile(fptr, dtype='i4', count=1)[0]
            fptr.seek(1024, 0)
            ones = np.fromfile(fptr, dtype='i4', count=num_data)
            multi = np.fromfile(fptr, dtype='i4', count=num_data)
        pdict['num_data'] = num_data
        pdict['ones_accum'] = np.cumsum(ones)
        pdict['multi_accum'] = np.cumsum(multi)

    def _parse_h5header(self, pdict):
        with h5py.File(pdict['fname'], 'r') as fptr:
            pdict['num_data'] = fptr['place_ones'].shape[0]
            pdict['num_pix'] = fptr['num_pix'][0]

    def get_frame(self, num, **kwargs):
        """Get particular frame from file list
        The method determines the file with that frame number and reads it

        Arguments:
            num (int) - Frame number
            raw (bool, optional) - Whether to get unassembled frame

        Returns:
            Assembled or unassembled frame as a dense array
        """
        file_num = np.where(num < np.array([pdict['num_data'] for pdict in self.flist]))[0][0]
        #file_num = np.where(num < self.num_data_list)[0][0]
        if file_num == 0:
            frame_num = num
        else:
            frame_num = num - self.flist[file_num-1]['num_data']

        return self._read_frame(file_num, frame_num, **kwargs)

    def get_powder(self, raw=False, **kwargs):
        """Get virtual powder sum of all frames in file list

        Arguments:
            raw (bool, optional) - Whether to return unassembled powder sum

        Returns:
            Assembled or unassembled powder sum as a dense array
        """
        if self.multiple_geom:
            raise ValueError('Powder sum unreasonable with multiple geometries')
        powder = np.zeros((self.flist[0]['num_pix'],), dtype='f8')

        for pdict in self.flist:
            if pdict['is_hdf5']:
                with h5py.File(pdict['fname'], 'r') as fptr:
                    place_ones = np.hstack(fptr['place_ones'][:])
                    place_multi = np.hstack(fptr['place_multi'][:])
                    count_multi = np.hstack(fptr['count_multi'][:])
            else:
                with open(pdict['fname'], 'rb') as fptr:
                    num_data = np.fromfile(fptr, dtype='i4', count=1)[0]
                    fptr.seek(1024, 0)
                    ones = np.fromfile(fptr, dtype='i4', count=num_data)
                    multi = np.fromfile(fptr, dtype='i4', count=num_data)
                    place_ones = np.fromfile(fptr, dtype='i4', count=ones.sum())
                    place_multi = np.fromfile(fptr, dtype='i4', count=multi.sum())
                    count_multi = np.fromfile(fptr, dtype='i4', count=multi.sum())

            np.add.at(powder, place_ones, 1)
            np.add.at(powder, place_multi, count_multi)

        powder *= self.flist[0]['geom'].unassembled_mask
        if not raw:
            powder = self._assemble_frame(powder, 0, **kwargs)
        return powder

    def _read_frame(self, file_num, frame_num, raw=False, sparse=False, **kwargs):
        pdict = self.flist[file_num]
        if pdict['is_hdf5']:
            po, pm, cm = self._read_h5frame(pdict, frame_num)
        else:
            po, pm, cm = self._read_binaryframe(pdict, frame_num)
            
        if sparse:
            return po, pm, cm
        frame = np.zeros(pdict['num_pix'], dtype='i4')
        np.add.at(frame, po, 1)
        np.add.at(frame, pm, cm)
        frame *= pdict['geom'].unassembled_mask
        if not raw:
            frame = self._assemble_frame(frame, file_num, **kwargs)
        return frame

    def _read_h5frame(self, pdict, frame_num):
        with h5py.File(pdict['fname'], 'r') as fptr:
            place_ones = fptr['place_ones'][frame_num]
            place_multi = fptr['place_multi'][frame_num]
            count_multi = fptr['count_multi'][frame_num]
        return place_ones, place_multi, count_multi

    def _read_binaryframe(self, pdict, frame_num):
        with open(pdict['fname'], 'rb') as fptr:
            num_data = np.fromfile(fptr, dtype='i4', count=1)[0]

            accum = [pdict['ones_accum'], pdict['multi_accum']]
            offset = [0, 0]
            size = [0, 0]

            if frame_num == 0:
                size = [accum[0][frame_num], accum[1][frame_num]]
            else:
                offset = [accum[0][frame_num-1], accum[1][frame_num-1]]
                size[0] = accum[0][frame_num] - accum[0][frame_num - 1]
                size[1] = accum[1][frame_num] - accum[1][frame_num - 1]

            fptr.seek(1024 + num_data*8 + offset[0]*4, 0)
            place_ones = np.fromfile(fptr, dtype='i4', count=size[0])
            fptr.seek(1024 + num_data*8 + accum[0][-1]*4 + offset[1]*4, 0)
            place_multi = np.fromfile(fptr, dtype='i4', count=size[1])
            fptr.seek(1024 + num_data*8 + accum[0][-1]*4 + accum[1][-1]*4 + offset[1]*4, 0)
            count_multi = np.fromfile(fptr, dtype='i4', count=size[1])
        return place_ones, place_multi, count_multi

    def _assemble_frame(self, data, num, fresh=False, zoomed=False):
        geom = self.flist[num]['geom']
        if fresh:
            img = np.zeros(geom.frame_shape, dtype=data.dtype)
        else:
            mask = 1-self._assembled_masks[num]
            img = ma.masked_array(np.zeros(mask.shape, dtype='i4'), mask=mask)
        np.add.at(img, (geom.x, geom.y), data)
        if zoomed:
            b = self.flist[num]['zoom_bounds']
            return img[b[0]:b[1], b[2]:b[3]]
        else:
            return img
