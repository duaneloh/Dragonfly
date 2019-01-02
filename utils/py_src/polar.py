'''Module containing class to convert frames to polar representation and derivatives'''

from __future__ import print_function
import numpy as np

class PolarConverter(object):
    """ Converts image to polar representation and calculates angular correlation

    The class is initialized by supplying the detector geometry in 2D as well as
    a mask file. The polar representation is a binned form and the binning
    parameters can also be specified.

    Methods:
        compute_polar(input_frame)
        compute_ang_corr(polar_arr)
        convert(input_frame)

    __init__ arguments
        x, y (array) - x and y coordinates of every pixel
        mask (array, int) - Representing whether the pixel
                            should be included (1) or not (0)
        r_min, rmax (float, optional) - Min and Max radii for angular averaging
        delta_r (float, optional) - Radial bin size
        delta_ang (float, optional) - Angular bin size in degrees
    """
    def __init__(self, x, y, mask, r_min=5, r_max=60, delta_r=2., delta_ang=10.):
        self.x = x.flatten()
        self.y = y.flatten()
        self.mask = mask.flatten()
        self.delta_r = delta_r
        self.delta_ang = delta_ang
        self.r_min = int(r_min/delta_r)
        self.r_max = int(np.ceil(r_max/delta_r))

        self.first_pass = True
        self.angs = self.sel_pixels = self.ang_max = self.rad_max = None
        self.polar_indices = self.polar_count = None

    def compute_indices(self):
        """Compute the angular and radial bins using the first encountered frame.
        """
        ang = np.mod(np.arctan2(self.x, self.y), np.pi) / (np.pi*self.delta_ang/180.)
        self.angs = (ang - ang.min()).astype('i4')

        raw_rads = np.sqrt(self.x*self.x + self.y*self.y)
        bin_rads = (raw_rads/self.delta_r).astype('i4')
        self.sel_pixels = (raw_rads <= self.r_max) & (raw_rads >= self.r_min)

        self.ang_max, self.rad_max = (self.angs.max()+1, bin_rads.max()+1)
        self.polar_indices = [a*self.ang_max+b for a, b in zip(bin_rads, self.angs)]
        self.polar_count = np.zeros(self.rad_max*self.ang_max)
        np.add.at(self.polar_count, self.polar_indices, self.mask)
        self.first_pass = False

    def convert(self, input_frame, method='ang_corr_normed'):
        """Converts input diffraction pattern according to method.

        Arguments:
            input_frame (array) - Input diffraction pattern
            method (string, optional) - Conversion method

        Possible methods:
            'raw' - Detector pixels in radius range as a 1D array
            'polar' - Polar representation from compute_polar()
            'ang_corr' - Row-wise FFT magnitudes of polar representation from
                         compute_ang_corr()

        Returns:
            Converted array
        """
        if self.first_pass:
            self.compute_indices()

        if method == 'ang_corr_normed':
            return self.compute_ang_corr(input_frame, normed=True)
        elif method == 'ang_corr':
            return self.compute_ang_corr(input_frame, normed=False)
        elif method == 'polar':
            return self.compute_polar(input_frame)
        elif method == 'polar_normed':
            return self.compute_polar(input_frame, normed=True)
        elif method == 'raw':
            return self.compute_raw(input_frame)
        elif method == 'raw_normed':
            return self.compute_raw(input_frame, normed=True)
        print('Unknown method string: %s'%method)
        return None

    def compute_raw(self, input_frame, normed=False):
        """Get pixels within supplied radius range

        Arguments:
            input_frame (array) - Input diffraction pattern

        Returns:
            1D array of pixel values within given radius range
        """
        if self.first_pass:
            self.compute_indices()
        data = input_frame[self.sel_pixels]
        if normed and data.mean() > 0:
            return data/data.mean()
        return data

    def compute_polar(self, input_frame, normed=True):
        """Converts given input diffraction pattern into a polar representation

        Arguments:
            input_frame (array) - Normally represented data frame

        Returns:
            polar_arr (array) - Polar representation of input_frame
        """
        if self.first_pass:
            self.compute_indices()

        polar_arr = np.zeros(self.rad_max*self.ang_max)
        np.add.at(polar_arr, self.polar_indices, input_frame.flatten()*self.mask)
        polar_arr[self.polar_count > 0] /= self.polar_count[self.polar_count > 0]
        polar_arr = polar_arr.reshape(self.rad_max, -1)[self.r_min:self.r_max]
        if normed:
            return polar_arr / polar_arr.mean()
        return polar_arr

    def compute_ang_corr(self, input_frame, normed=True, ang_max=10):
        """Compute the angular correlation from the polar representation of given pattern

        Arguments:
            polar_arr (array) - Polar data array (usually output of convert())
            normed (bool, optional) - Whether to normalize Fourier transform in each radial bin
            ang_max (float, optional) - How many Fourier components to keep

        Returns:
            ang_corr (array) - Angular correlations for each input bin
        """
        polar_arr = self.compute_polar(input_frame)
        ang_corr = np.array([arr - arr.mean() for arr in polar_arr])
        temp = []
        for corr in ang_corr:
            if normed:
                norm = np.linalg.norm(corr)
                if norm > 0.:
                    temp.append(np.absolute(np.fft.fft(corr/norm))[1:ang_max])
                else:
                    temp.append(np.zeros(ang_max-1))
            else:
                temp.append(np.absolute(np.fft.fft(corr))[1:ang_max])
        return np.array(temp)
