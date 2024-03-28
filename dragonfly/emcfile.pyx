'''Module containing classes to interface with EMC files'''

from __future__ import print_function
import sys
import os
import os.path as op
import time
import numpy as np
import pandas
import h5py

from . cimport emcfile as c_dset
from .emcfile cimport CDataset
from .detector cimport CDetector
from libc.stdio cimport FILE, fopen, fclose, fread, fseek
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcpy, memcpy

cdef class CDataset:
    def __init__(self, fname=None, CDetector det=None):
        self.dset = <dataset*> calloc(1, sizeof(dataset))
        self.dset.det = det.det if det is not None else NULL
        if fname is not None:
            self.parse(fname)

    def parse(self, fname):
        if self.dset.det == NULL:
            raise AttributeError('Set detector before parsing')
        self.dset.fname = <char*> malloc(len(fname)+1)
        strcpy(self.dset.fname, bytes(fname, 'utf-8'))

        c_dset.parse_dataset(self.dset.fname, self.dset.det, self.dset)

    def append(self, CDataset next_dset):
        cdef c_dset.dataset *curr = self.dset
        while curr.next != NULL:
            curr = curr.next
        curr.next = next_dset.dset

    def free(self):
        if self.dset.ones != NULL: free(self.dset.ones)
        if self.dset.multi != NULL: free(self.dset.multi)
        if self.dset.place_ones != NULL: free(self.dset.place_ones)
        if self.dset.place_multi != NULL: free(self.dset.place_multi)
        if self.dset.count_multi != NULL: free(self.dset.count_multi)
        if self.dset.ones_accum != NULL: free(self.dset.ones_accum)
        if self.dset.multi_accum != NULL: free(self.dset.multi_accum)

    def __iter__(self):
        curr = self
        while curr is not None:
            yield curr
            curr = curr.next

    @property
    def fname(self): return (<bytes> self.dset.fname).decode()
    @property
    def num_data(self): return self.dset.num_data
    @property
    def num_pix(self): return self.dset.num_pix
    @property
    def ftype(self): return ['sparse', 'dense_integer', 'dense_double'][self.dset.ftype]
    @property
    def det(self):
        if self.dset.det == NULL:
            return
        retval = CDetector()
        retval.det = self.dset.det
        return retval
    @det.setter
    def det(self, CDetector det): self.dset.det = det.det
    @property
    def next(self):
        if self.dset.next == NULL:
           return
        retval = CDataset()
        retval.dset = self.dset.next
        return retval

    @property
    def ones(self): return np.asarray(<int[:self.num_data]>self.dset.ones, dtype='i4')
    @property
    def multi(self): return np.asarray(<int[:self.num_data]>self.dset.multi, dtype='i4')
    @property
    def place_ones(self): return np.asarray(<int[:self.ones_total]>self.dset.place_ones, dtype='i4')
    @property
    def place_multi(self): return np.asarray(<int[:self.multi_total]>self.dset.place_multi, dtype='i4')
    @property
    def count_multi(self): return np.asarray(<int[:self.multi_total]>self.dset.count_multi, dtype='i4')
    @property
    def ones_total(self): return self.dset.ones_total
    @property
    def multi_total(self): return self.dset.multi_total
    @property
    def ones_accum(self): return np.asarray(<long[:self.num_data]>self.dset.ones_accum, dtype='i8')
    @property
    def multi_accum(self): return np.asarray(<long[:self.num_data]>self.dset.multi_accum, dtype='i8')

class EMCReader():
    """EMC file reader

    Provides access to assembled or raw frames given a list of .emc filenames

    __init__ arguments:
        photons_list - Path or sequence of paths to emc files. If single file, pass as [fname]
        det_list - Single or list of Detector objects.
        det_mapping (list, optional) - Mapping from photons_list to det_list

    If there is only one entry in det_list, all emc files are assumed to use \
    that detector. Otherwise, a mapping must be provided. \
    The mapping is a list of the same length as photons_list with entries \
    giving indices in det_list for the corresponding emc file.

    Methods:
        get_frame(num, raw=False, sparse=False, zoomed=False, sym=False)
        get_powder(raw=False, zoomed=False, sym=False)
        get_blacklist(fname, sel_string=None)
    """
    def __init__(self, photons_list, det_list, dset_list=None, det_mapping=None):
        '''Create object for given photons file list and detector list
        
        One can also pass single file names and detector objects if needed
        '''
        # Convert to lists if singleton arguments
        if hasattr(photons_list, 'strip') or not hasattr(photons_list, '__getitem__'):
            photons_list = [photons_list]
        if not hasattr(det_list, '__getitem__'):
            det_list = [det_list]
        # Create dictionaries of photons file information
        self.flist = [{'fname': fname} for fname in photons_list]
        num_files = len(photons_list)

        self.multiple_det = False
        if len(det_list) == 1:
            for i in range(num_files):
                self.flist[i]['det'] = det_list[0]
        else:
            try:
                for i in range(num_files):
                    self.flist[i]['det'] = det_list[det_mapping[i]]
                self.multiple_det = True
            except TypeError:
                print('Need mapping if multiple geometries are provided')
                raise

        if dset_list is not None:
            if hasattr(dset_list, 'strip') or not hasattr(dset_list, '__getitem__'):
                dset_list = [dset_list]

            if len(dset_list) != len(photons_list):
                raise ValueError('dset_list must be same length as photons_list')
            self._dset_list = []
            for dset in dset_list:
                if dset is not None:
                    self._dset_list.append(dset)
                else:
                    self._dset_list.append(None)
        else:
            self._dset_list = [None for _ in range(len(photons_list))]

        self._parse_headers()

    def get_frame(self, num, **kwargs):
        """Get particular frame from file list
        The method determines the file with that frame number and reads it

        Arguments:
            num (int) - Frame number
        Keyword arguments:
            raw (bool) - Whether to get unassembled frame (False)
            sparse (bool) - Whether to return sparse data (False)
            zoomed (bool) - Whether to zoom assembled frame to non-masked region (False)
            sym (bool) - Whether to centro-symmetrize assembled frame (False)

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

    def get_powder(self, raw=False, verbose=False, **kwargs):
        """Get virtual powder sum of all frames in file list

        Keyword arguments:
            raw (bool) - Whether to return unassembled powder sum (False)
            zoomed (bool) - Whether to zoom assembled frame to non-masked region (False)
            sym (bool) - Whether to centro-symmetrize assembled frame (False)

        Returns:
            Assembled or unassembled powder sum as a dense array
        """
        if self.multiple_det:
            raise ValueError('Powder sum unreasonable with multiple geometries')
        powder = np.zeros((self.flist[0]['num_pix'],), dtype='f8')

        for pdict in self.flist:
            if pdict['frame_type'] == 0:
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
            else:
                if pdict['is_hdf5']:
                    with h5py.File(pdict['fname'], 'r') as fptr:
                        powder += fptr[pdict['dset_name']][:].sum(0).ravel()
                else:
                    with open(pdict['fname'], 'rb') as fptr:
                        num_data = np.fromfile(fptr, dtype='i4', count=1)[0]
                        fptr.seek(1024, 0)
                        if pdict['frame_type'] == 1:
                            for _ in range(num_data):
                                powder += np.fromfile(fptr, dtype='i4', count=pdict['num_pix'])
                        elif pdict['frame_type'] == 2:
                            for _ in range(num_data):
                                powder += np.fromfile(fptr, dtype='f8', count=pdict['num_pix'])
            if verbose:
                print(pdict['fname'], pdict['num_data'])

        if not raw:
            powder = self.flist[0]['det'].assemble_frame(powder, **kwargs)
        return powder

    def _parse_headers(self):
        for i, pdict in enumerate(self.flist):
            pdict['dset_name'] = self._dset_list[i]
            pdict['is_hdf5'] = h5py.is_hdf5(pdict['fname'])
            pdict['parsed'] = False
            if pdict['is_hdf5']:
                self._parse_h5header(pdict)
            else:
                self._parse_binaryheader(pdict)

            if pdict['num_pix'] != len(pdict['det'].x):
                sys.stderr.write(
                    'Warning: num_pix for %s is different (%d vs %d)\n' %
                    (pdict['fname'], pdict['num_pix'], len(pdict['det'].x)))
            if i > 0:
                pdict['num_data'] += self.flist[i-1]['num_data']
        self.num_frames = self.flist[len(self.flist)-1]['num_data']
        self.blacklist = np.zeros(self.num_frames, dtype='u1')
        self.num_blacklist = 0

    @staticmethod
    def _parse_binaryheader(pdict):
        with open(pdict['fname'], 'rb') as fptr:
            num_data = np.fromfile(fptr, dtype='i4', count=1)[0]
            pdict['num_pix'] = np.fromfile(fptr, dtype='i4', count=1)[0]
            pdict['frame_type'] = np.fromfile(fptr, dtype='i4', count=1)[0]
            fptr.seek(1024, 0)
            ones = np.fromfile(fptr, dtype='i4', count=num_data)
            multi = np.fromfile(fptr, dtype='i4', count=num_data)
        pdict['num_data'] = num_data
        pdict['ones_accum'] = np.cumsum(ones)
        pdict['multi_accum'] = np.cumsum(multi)

    @staticmethod
    def _parse_h5header(pdict):
        with h5py.File(pdict['fname'], 'r') as fptr:
            if pdict['dset_name'] is None:
                pdict['frame_type'] = 0
                pdict['num_data'] = fptr['place_ones'].shape[0]
                pdict['num_pix'] = np.prod(fptr['num_pix'][()])
            else:
                pdict['frame_type'] = 1
                dset = fptr[pdict['dset_name']]
                pdict['num_data'] = dset.shape[0]
                pdict['num_pix'] = np.prod(dset.shape[1:])

    @staticmethod
    def _read_h5frame(pdict, frame_num):
        with h5py.File(pdict['fname'], 'r') as fptr:
            if pdict['frame_type'] == 0:
                place_ones = fptr['place_ones'][frame_num]
                place_multi = fptr['place_multi'][frame_num]
                count_multi = fptr['count_multi'][frame_num]
                return place_ones, place_multi, count_multi
            else:
                return fptr[pdict['dset_name']][frame_num].ravel()

    @staticmethod
    def _read_binaryframe(pdict, frame_num):
        fptr = open(pdict['fname'], 'rb')
        num_data = np.fromfile(fptr, dtype='i4', count=1)[0]

        if pdict['frame_type'] == 0:
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
            fptr.seek(1024 + num_data*8 + accum[0][num_data-1]*4 + offset[1]*4, 0)
            place_multi = np.fromfile(fptr, dtype='i4', count=size[1])
            fptr.seek(1024 + num_data*8 + accum[0][num_data-1]*4 + accum[1][num_data-1]*4 + offset[1]*4, 0)
            count_multi = np.fromfile(fptr, dtype='i4', count=size[1])
            fptr.close()

            return place_ones, place_multi, count_multi
        elif pdict['frame_type'] == 1:
            fptr.seek(1024 + frame_num*pdict['num_pix']*4, 0)
            frame = np.fromfile(fptr, dtype='i4', count=pdict['num_pix'])
            fptr.close()
            return frame
        elif pdict['frame_type'] == 2:
            fptr.seek(1024 + frame_num*pdict['num_pix']*8, 0)
            frame = np.fromfile(fptr, dtype='f8', count=pdict['num_pix'])
            fptr.close()
            return frame

    def _read_frame(self, file_num, frame_num, raw=False, sparse=False, **kwargs):
        pdict = self.flist[file_num]
        if pdict['is_hdf5']:
            frame_data = self._read_h5frame(pdict, frame_num) # pylint: disable=invalid-name
        else:
            frame_data = self._read_binaryframe(pdict, frame_num) # pylint: disable=invalid-name

        if pdict['frame_type'] == 0:
            po, pm, cm = frame_data
            if sparse:
                return po, pm, cm
            frame = np.zeros(pdict['num_pix'], dtype='i4')
            np.add.at(frame, po, 1)
            np.add.at(frame, pm, cm)
        else:
            if sparse:
                raise ValueError('Asking for sparse data when the file contains dense frames')
            frame = frame_data

        if not raw:
            frame = pdict['det'].assemble_frame(frame, **kwargs)
        return frame

class EMCWriter(object):
    """EMC file writer class

    Provides interface to write dense integer photon count data to an emc file

    __init__ arguments:
        out_fname (string) - Output filename
        num_pix (int) - Number of pixels in dense frame

    The number of pixels is saved to the header and serves as a check since the
    sparse format is in reference to a detector file.

    Methods:
        write_frame(frame, fraction=1.)
        write_sparse_frame(place_ones, place_multi, count_multi)
        finish_write()

    The typical usage is as follows:

    .. code-block:: python

       with EMCWriter('photons.emc', num_pix) as emc:
           for i in range(num_frames):
               emc.write_frame(frame[i].ravel())
    """

    def __init__(self, out_fname, num_pix, hdf5=True):
        self.h5_output = hdf5

        self.out_fname = out_fname
        if self.h5_output:
            print('Writing HDF5 emc file to', out_fname)
        else:
            print('Writing binary emc file to', out_fname)
        self.num_data = 0
        self.num_pix = num_pix
        self.mean_count = 0.
        self.ones = []
        self.multi = []
        self._init_file()

    def __enter__(self):
        return self

    def __exit__(self, etype, val, traceback):
        self.finish_write()

    def _init_file(self):
        if self.h5_output:
            self._h5f = h5py.File(self.out_fname, 'w')
            self._h5f['num_pix'] = [self.num_pix]

            vlentype = h5py.special_dtype(vlen=np.int32)
            self._h5f.create_dataset('place_ones', (0,), maxshape=(None,),
                                     chunks=(1,), dtype=vlentype)
            self._h5f.create_dataset('place_multi', (0,), maxshape=(None,),
                                     chunks=(1,), dtype=vlentype)
            self._h5f.create_dataset('count_multi', (0,), maxshape=(None,),
                                     chunks=(1,), dtype=vlentype)
            self._fptrs = []
        else:
            counter = 0
            while True:
                folder = op.dirname(self.out_fname)
                basename = '.' + op.splitext(op.basename(self.out_fname))[0]
                temp_fnames = [op.join(folder, basename+fname) + str(os.getpid()+counter)
                               for fname in ['.po.', '.pm.', '.cm.']]
                if op.exists(temp_fnames[0]):
                    counter += 1
                else:
                    break

            self._fptrs = [open(fname, 'wb') for fname in temp_fnames]

    def finish_write(self, header_nums=None):
        """Cleanup and close emc file

        This function writes the header and appends the temporary files.
        It then deletes those temp files. This function should be run before
        the script is exited.
        """
        for fptr in self._fptrs:
            fptr.close()
        if self.h5_output:
            self._h5f.close()

        if self.num_data == 0:
            print('No frames to write')
            for fptr in self._fptrs:
                os.system('rm ' + fptr.name)
            return

        self.mean_count /= self.num_data
        print('num_data = %d, mean_count = %.4e' % (self.num_data, self.mean_count))

        if not self.h5_output:
            ones_arr = np.asarray(self.ones)
            multi_arr = np.asarray(self.multi)

            fptr = open(self.out_fname, 'wb')
            header = np.zeros((256), dtype='i4')
            header[0] = self.num_data
            header[1] = self.num_pix
            if header_nums is not None:
                if len(header_nums) <= 254:
                    header[2:len(header_nums)+2] = np.array(header_nums).astype('i4')
                else:
                    header[2:] = np.array(header_nums).astype('i4')[:254]
            header.tofile(fptr)
            ones_arr.astype('i4').tofile(fptr)
            multi_arr.astype('i4').tofile(fptr)
            fptr.close()
            for fptr in self._fptrs:
                os.system('cat ' + fptr.name + ' >> ' + self.out_fname)
                os.system('rm ' + fptr.name)

    def write_frame(self, frame, fraction=1., partition=1):
        """Write given frame to the file

        Using temporary files, the sparsified version of the input is written.

        Arguments:
            frame (int array) - 1D dense array with photon counts in each pixel
            fraction (float, optional) - What fraction of photons to write
            partition (int, optional) - Partition frame into N sub-frames

        If fraction is less than 1, then each photon is written randomly with \
        the probability = fraction. by default, all photons are written. This \
        option is useful for performing tests with lower photons/frame.
        """
        if len(frame.shape) != 1 or not np.issubdtype(frame.dtype, np.integer):
            raise ValueError('write_frame needs 1D array of integers: '+
                             str(frame.shape)+' '+str(frame.dtype))

        place_ones = np.where(frame == 1)[0]
        place_multi = np.where(frame > 1)[0]
        count_multi = frame[place_multi]

        if fraction < 1. and partition > 1:
            print('Can either split or reduce data frame')
            return
        elif partition > 1:
            sel_ones = (np.random.random(len(place_ones))*int(partition)).astype('i4')
            sel_multi = (np.random.random(count_multi.sum())*int(partition)).astype('i4')
            sum_count_multi = count_multi.cumsum()
            for i in range(int(partition)):
                sp_count_multi = np.array([a.sum() for a in np.split(sel_multi == i, sum_count_multi)])
                sp_count_multi = sp_count_multi[:len(sp_count_multi)-1]
                sp_place_multi = place_multi[sp_count_multi > 0]
                sp_count_multi = sp_count_multi[sp_count_multi > 0]
                self._update_file(place_ones[sel_ones == i], sp_place_multi, sp_count_multi)
        elif fraction < 1.:
            sel = (np.random.random(len(place_ones)) < fraction)
            place_ones = place_ones[sel]
            sel = (np.random.random(count_multi.sum()) < fraction)
            count_multi = np.array([a.sum() for a in np.split(sel, count_multi.cumsum())])
            count_multi = count_multi[:len(count_multi)-1]
            place_multi = place_multi[count_multi > 0]
            count_multi = count_multi[count_multi > 0]
            self._update_file(place_ones, place_multi, count_multi)
        else:
            self._update_file(place_ones, place_multi, count_multi)

    def write_sparse_frame(self, place_ones, place_multi, count_multi):
        """Write sparse frame to file

        Arguments:
            place_ones (int array) - List of pixel numbers with 1 photon
            place_multi (int array) - List of pixel numbers with moe than 1 photon
            count_multi (int array) - Number of photons in the place_multi pixels

        len(place_multi) == len(count_multi)
        """
        if len(place_multi) != len(count_multi):
            raise ValueError('place_multi and count_multi should have equal lengths')
        if not (np.issubdtype(place_ones.dtype, np.integer)
                and np.issubdtype(place_multi.dtype, np.integer)
                and np.issubdtype(count_multi.dtype, np.integer)):
            raise ValueError('Arrays should be of integer type')

        self._update_file(place_ones, place_multi, count_multi)

    def _update_file(self, place_ones, place_multi, count_multi):
        self.num_data += 1
        self.mean_count += len(place_ones) + count_multi.sum()
        self.ones.append(len(place_ones))
        self.multi.append(len(place_multi))

        if self.h5_output:
            self._h5f['place_ones'].resize((self.num_data,))
            self._h5f['place_ones'][self.num_data-1] = place_ones.astype(np.int32)
            self._h5f['place_multi'].resize((self.num_data,))
            self._h5f['place_multi'][self.num_data-1] = place_multi.astype(np.int32)
            self._h5f['count_multi'].resize((self.num_data,))
            self._h5f['count_multi'][self.num_data-1] = count_multi.astype(np.int32)
        else:
            place_ones.astype(np.int32).tofile(self._fptrs[0])
            place_multi.astype(np.int32).tofile(self._fptrs[1])
            count_multi.astype(np.int32).tofile(self._fptrs[2])

