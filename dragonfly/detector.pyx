'''Module containing detector class'''

import sys
import os
import numpy as np
from numpy import ma
import pandas
import h5py

from . cimport detector as c_det
from .detector cimport Detector
from libc.stdlib cimport malloc, calloc
from libc.string cimport strcpy

cdef class Detector:
    """Dragonfly detector cython class

    The detector file format is specified in github.com/duaneloh/Dragonfly/wiki
    This class reads the file and provides numpy arrays which can be used for
    further processing.

    __init__ arguments (optional):
        fname (string) - Path to detector file to populate attributes
        mask_flag (bool) - Whether to read the mask column for each pixel
        keep_mask_1 (bool) - Whether to consider mask=1 pixels as good

    Methods:
        parse(fname, mask_flag=False, keep_mask_1=True)
        write(fname)
        assemble_frame(data, zoomed=False, sym=False)
        calc_from_coords()
        remask(qradius)

    On parsing, it produces the following numpy arrays (each of length num_pix)

    Attributes:
        self.qvals - Voxel space coordinates (origin at (0,0,0))
        self.cx, self.cy - Floating point 2D coordinates (origin at (0,0))
        self.x, self.y - Integer and shifted 2D coordinates (corner at (0,0))
        self.mask - Assembled mask
        self.raw_mask - Unassembled mask as stored in detector file
        self.unassembled_mask - Unassembled mask (1=good, 0=bad)
    """
    def __init__(self, fname=None, **kwargs):
        self.det = <c_det.detector*> calloc(1, sizeof(c_det.detector))
        #self._sym_shape = None
        if fname is not None:
            self.parse(fname, **kwargs)

    def parse(self, fname, mask_flag=True, keep_mask_1=True):
        """ Parse Dragonfly detector from file

        File can either be in the HDF5 or ASCII format
        """
        self.det.fname = <char*> malloc(len(fname)+1)
        strcpy(self.det.fname, bytes(fname, 'utf-8'))
        if h5py.is_hdf5(fname):
            self._parse_h5det(mask_flag, keep_mask_1)
        elif os.path.splitext(fname)[1] == '.h5':
            fheader = np.fromfile(fname, '=c', count=8)
            if fheader == chr(137)+'HDF\r\n'+chr(26)+'\n':
                self._parse_h5det(mask_flag, keep_mask_1)
            else:
                self._parse_asciidet(mask_flag, keep_mask_1)
        else:
            self._parse_asciidet(mask_flag, keep_mask_1)

    def _check_header(self):
        with open(self.fname, 'r') as fptr:
            line = fptr.readline().rstrip().split()
        if len(line) != 1:
            self.det.detd = float(line[1])
            self.det.ewald_rad = float(line[2])
        else:
            raise ValueError('Need 3 values on header line: num_pix, detd_pix, ewald_rad_vox')

    def _parse_asciidet(self, mask_flag, keep_mask_1):
        print('Parsing ASCII detector file')
        self._check_header()
        dframe = pandas.read_csv(
            self.fname,
            delim_whitespace=True, skiprows=1, engine='c', header=None,
            names=['qx', 'qy', 'qz', 'corr', 'mask'],
            dtype={'qx':'f8', 'qy':'f8', 'qz':'f8', 'corr':'f8', 'mask':'u1'})
        qx, qy, qz, np_corr = tuple([np.array(dframe[key]) # pylint: disable=C0103
                                       for key in ['qx', 'qy', 'qz', 'corr']])
        self.det.num_pix = qx.shape[0]

        cdef int t, d
        cdef double[:,:] qvals = np.copy(np.array([qx, qy, qz]).T)
        cdef double[:] corr = np_corr
        cdef uint8_t[:] raw_mask = np.array(dframe['mask']).astype('u1')
        self.det.qvals = <double*> malloc(self.num_pix * 3 * sizeof(double))
        self.det.corr = <double*> malloc(self.num_pix * sizeof(double))
        self.det.raw_mask = <uint8_t*> malloc(self.num_pix * sizeof(uint8_t))

        for t in range(self.num_pix):
            self.det.corr[t] = corr[t]
            self.det.raw_mask[t] = raw_mask[t]
            for d in range(3):
                self.det.qvals[t*3 + d] = qvals[t, d]

        self._process_det(mask_flag, keep_mask_1)

    def _parse_h5det(self, mask_flag, keep_mask_1):
        print('Parsing HDF5 detector file')

        fptr = h5py.File(self.fname, 'r')
        qx, qy, qz = fptr['qx'][:], fptr['qy'][:], fptr['qz'][:]
        cdef double[:] corr = fptr['corr'][:]
        cdef uint8_t[:] raw_mask = fptr['mask'][:].astype('u1')
        self.det.detd = fptr['detd'][()]
        self.det.ewald_rad = fptr['ewald_rad'][()]
        fptr.close()

        self.det.num_pix = qx.shape[0]
        cdef double[:,:] qvals = np.copy(np.array([qx, qy, qz]).T)

        self.det.qvals = <double*> malloc(self.num_pix * 3 * sizeof(double))
        self.det.corr = <double*> malloc(self.num_pix * sizeof(double))
        self.det.raw_mask = <uint8_t*> malloc(self.num_pix * sizeof(uint8_t))

        cdef int t, d
        for t in range(self.num_pix):
            self.det.corr[t] = corr[t]
            self.det.raw_mask[t] = raw_mask[t]
            for d in range(3):
                self.det.qvals[t*3 + d] = qvals[t, d]

        self._process_det(mask_flag, keep_mask_1)

    def _process_det(self, mask_flag, keep_mask_1):
        if mask_flag:
            mask = np.asarray(<uint8_t[:self.num_pix]>self.det.raw_mask)
            if keep_mask_1:
                mask[mask == 1] = 0 # To keep both 0 and 1
                mask = mask // 2 # To keep both 0 and 1
            else:
                mask[mask == 2] = 1 # To keep only mask==0
            mask = 1 - mask
        else:
            self.raw_mask = np.zeros(self.corr.shape, dtype='u1')
            mask = np.ones(self.corr.shape, dtype='u1')

        cdef int t
        cdef double minx = 1.e10
        cdef double miny = 1.e10
        self.det.cx = <double*> malloc(self.num_pix * sizeof(double))
        self.det.cy = <double*> malloc(self.num_pix * sizeof(double))
        self.det.x = <int*> malloc(self.num_pix * sizeof(int))
        self.det.y = <int*> malloc(self.num_pix * sizeof(int))
        self.det.mask = <uint8_t*> malloc(self.num_pix * sizeof(uint8_t))

        sign = self.qvals[:,2].mean() > 0
        for t in range(self.num_pix):
            if sign > 0:
                self.det.cx[t] = self.det.qvals[t*3 + 0] * self.detd / (self.ewald_rad - self.det.qvals[t*3 + 2])
                self.det.cy[t] = self.det.qvals[t*3 + 1] * self.detd / (self.ewald_rad - self.det.qvals[t*3 + 2])
            else:
                self.det.cx[t] = self.det.qvals[t*3 + 0] * self.detd / (self.ewald_rad + self.det.qvals[t*3 + 2])
                self.det.cy[t] = self.det.qvals[t*3 + 1] * self.detd / (self.ewald_rad + self.det.qvals[t*3 + 2])
            if self.det.cx[t] < minx:
                minx = self.det.cx[t]
            if self.det.cy[t] < miny:
                miny = self.det.cy[t]

        for t in range(self.num_pix):
            self.det.x[t] = int(np.round(self.det.cx[t] - minx))
            self.det.y[t] = int(np.round(self.det.cy[t] - miny))
            self.det.mask[t] = mask[t]

        self._init_assem()

    def _init_assem(self):
        '''Calculate attributes given self.x and self.y'''
        fshape = self.frame_shape
        
        cdef int t
        self.det.assembled_mask = <uint8_t*> calloc(fshape[0]*fshape[1], sizeof(uint8_t))
        for t in range(self.num_pix):
            self.det.assembled_mask[self.det.x[t]*fshape[1] + self.det.y[t]] = self.det.mask[t]
        #self.mask = np.sign(self.mask)

        #xsel = self.x[mask.astype(np.bool)]
        #ysel = self.y[mask.astype(np.bool)]
        #self.zoom_bounds = (xsel.min(), xsel.max()+1, ysel.min(), ysel.max()+1)

    @property
    def fname(self): return (<bytes> self.det.fname).decode()
    @property
    def num_pix(self): return self.det.num_pix
    @property
    def detd(self): return self.det.detd
    @property
    def ewald_rad(self): return self.det.ewald_rad
    @property
    def corr(self): return np.asarray(<double[:self.num_pix]>self.det.corr)
    @property
    def raw_mask(self): return np.asarray(<uint8_t[:self.num_pix]>self.det.raw_mask)
    @property
    def qvals(self): return np.asarray(<double[:3*self.num_pix]>self.det.qvals).reshape(-1, 3)
    @property
    def shape(self): return self.corr.shape if self.det.corr != NULL else None
    @property
    def frame_shape(self): return (self.x.max()+1, self.y.max()+1) if self.det.x != NULL else None
    @property
    def cx(self): return np.asarray(<double[:self.num_pix]>self.det.cx)
    @property
    def cy(self): return np.asarray(<double[:self.num_pix]>self.det.cy)
    @property
    def x(self): return np.asarray(<int[:self.num_pix]>self.det.x)
    @property
    def y(self): return np.asarray(<int[:self.num_pix]>self.det.y)
    @property
    def mask(self): return np.asarray(<uint8_t[:self.num_pix]>self.det.mask).astype('bool')
    @property
    def assembled_mask(self): 
        fshape = self.frame_shape
        return np.asarray(<uint8_t[:fshape[0],:fshape[1]]>self.det.assembled_mask).astype('bool')

