'''Module with EMC_writer class to save dense frames in EMC format'''

from __future__ import print_function
import os
import numpy as np
try:
    import h5py
    HDF5_MODE = True
except ImportError:
    HDF5_MODE = False

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
        out_folder = os.path.dirname(out_fname)
        self.h5_output = hdf5
        if hdf5 and not HDF5_MODE:
            print('Could not import h5py. Generating .emc output')
            out_fname = os.path.splitext(out_fname)[0] + '.emc.'
            self.h5_output = False

        self.out_fname = out_fname
        print('Writing emc file to', out_fname)
        self.num_data = 0
        self.num_pix = num_pix
        self.mean_count = 0.
        self.ones = []
        self.multi = []
        self._init_file(out_folder)

    def __enter__(self):
        return self

    def __exit__(self, etype, val, traceback):
        self.finish_write()

    def _init_file(self, out_folder):
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
            temp_fnames = [os.path.join(out_folder, fname) + str(os.getpid())
                           for fname in ['.po.', '.pm.', '.cm.']]
            self._fptrs = [open(fname, 'wb') for fname in temp_fnames]

    def finish_write(self):
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
                sp_count_multi = np.array([a.sum() for a in np.split(sel_multi == i, sum_count_multi)])[:-1]
                sp_place_multi = place_multi[sp_count_multi > 0]
                sp_count_multi = sp_count_multi[sp_count_multi > 0]
                self._update_file(place_ones[sel_ones == i], sp_place_multi, sp_count_multi)
        elif fraction < 1.:
            sel = (np.random.random(len(place_ones)) < fraction)
            place_ones = place_ones[sel]
            sel = (np.random.random(count_multi.sum()) < fraction)
            count_multi = np.array([a.sum() for a in np.split(sel, count_multi.cumsum())])[:-1]
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
            self._h5f['place_ones'][-1] = place_ones.astype(np.int32)
            self._h5f['place_multi'].resize((self.num_data,))
            self._h5f['place_multi'][-1] = place_multi.astype(np.int32)
            self._h5f['count_multi'].resize((self.num_data,))
            self._h5f['count_multi'][-1] = count_multi.astype(np.int32)
        else:
            place_ones.astype(np.int32).tofile(self._fptrs[0])
            place_multi.astype(np.int32).tofile(self._fptrs[1])
            count_multi.astype(np.int32).tofile(self._fptrs[2])
