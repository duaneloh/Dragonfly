'''Module with EMC_writer class to save dense frames in EMC format'''

from __future__ import print_function
import os
import numpy as np

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
        finish_write()

    The typical usage is as follows:
    emc = EMC_writer('photons.emc', num_pix)
    for i in range(num_frames):
        emc.write_frame(frame[i].flatten())
    emc.finish_write()
    """

    def __init__(self, out_fname, num_pix):
        out_folder = os.path.dirname(out_fname)
        temp_fnames = [os.path.join(out_folder, fname)
                       for fname in ['temp.po', 'temp.pm', 'temp.cm']]
        self._fptrs = [open(fname, 'wb') for fname in temp_fnames]

        self.out_fname = out_fname
        print('Writing emc file to', out_fname)
        self.num_data = 0
        self.num_pix = num_pix
        self.mean_count = 0.
        self.ones = []
        self.multi = []

    def finish_write(self):
        """Cleanup and close emc file
        This function writes the header and appends the temporary files.
        It then deletes those temp files. This function should be run before
        the script is exited.
        """
        for fptr in self._fptrs:
            fptr.close()

        if self.num_data == 0:
            for fptr in self._fptrs:
                os.system('rm ' + fptr.name)
            return

        self.mean_count /= self.num_data
        print('num_data = %d, mean_count = %.4e' % (self.num_data, self.mean_count))
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

    def write_frame(self, frame, fraction=1.):
        """Write given frame to the file
        Using temporary files, the sparsified version of the input is written.

        Arguments:
            frame (int array) - 1D dense array with photon counts in each pixel
            fraction (float, optional) - What fraction of photons to write
        If fraction is less than 1, then each photon is written randomly with
        the probability = fraction. by default, all photons are written. This
        option is useful for performing tests with lower photons/frame.
        """
        if len(frame.shape) != 1 or not np.issubdtype(frame.dtype, np.integer):
            raise ValueError('write_frame needs 1D array of integers: '+
                             str(frame.shape)+' '+str(frame.dtype))

        place_ones = np.where(frame == 1)[0]
        place_multi = np.where(frame > 1)[0]
        count_multi = frame[place_multi]

        if fraction < 1.:
            sel = (np.random.random(len(place_ones)) < fraction)
            place_ones = place_ones[sel]
            sel = (np.random.random(count_multi.sum()) < fraction)
            count_multi = np.array([a.sum() for a in np.split(sel, count_multi.cumsum())])[:-1]
            place_multi = place_multi[count_multi > 0]
            count_multi = count_multi[count_multi > 0]
        self.num_data += 1
        self.mean_count += len(place_ones) + count_multi.sum()
        self.ones.append(len(place_ones))
        self.multi.append(len(place_multi))

        place_ones.astype(np.int32).tofile(self._fptrs[0])
        place_multi.astype(np.int32).tofile(self._fptrs[1])
        count_multi.astype(np.int32).tofile(self._fptrs[2])
