import numpy as np
cimport numpy as np
import sys
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport emc 
cimport openmp
from detector cimport detector

cdef class detector:
	def __init__(self):
		self.det = <emc.detector*> PyMem_Malloc(sizeof(emc.detector))
		self.det.num_det = 0
		self.det.num_dfiles = 0
		self.det.mapping = [0]*1024

	def generate_detectors(self, config_fname, norm_flag=True, config_section='emc'):
		cdef char* c_config_fname = config_fname
		cdef char* c_config_section = config_section
		if self.det.num_det > 0: self.free_detector()
		qmax = emc.generate_detectors(c_config_fname, c_config_section, &self.det, int(norm_flag))
		assert qmax > 0.
		return qmax

	def parse_detector_list(self, flist, norm_flag=True):
		cdef char* c_flist = flist
		if self.det.num_det > 0: self.free_detector()
		qmax = emc.parse_detector_list(flist, &self.det, int(norm_flag))
		assert qmax > 0.
		return qmax

	def parse_detector(self, fname, norm_flag=True):
		cdef char* c_fname = fname
		if self.det.num_det > 0: self.free_detector()
		self.__init__()
		self.det.num_det = 1
		self.det.num_dfiles = 0
		qmax = emc.parse_detector(c_fname, self.det, int(norm_flag))
		assert qmax > 0.
		return qmax

	def free_detector(self):
		emc.free_detector(self.det)
		self.det = NULL

	def __del__(self):
		self.free_detector()

	@property
	def num_pix(self): return self.det.num_pix if self.det != NULL else None
	@property
	def rel_num_pix(self): return self.det.rel_num_pix if self.det != NULL else None
	@property
	def detd(self): return self.det.detd if self.det != NULL else None
	@property
	def ewald_rad(self): return self.det.ewald_rad if self.det != NULL else None
	@property
	def pixels(self): return np.asarray(<double[:4*self.num_pix]>self.det.pixels).reshape(-1,4) if self.det != NULL else None
	@property
	def mask(self): return np.asarray(<emc.uint8_t[:self.num_pix]>self.det.mask) if self.det != NULL else None

	# Only relevant for first detector in list
	@property
	def num_det(self): return self.det.num_det if self.det != NULL else None
	@property
	def num_dfiles(self): return self.det.num_dfiles if self.det != NULL else None
	@property
	def mapping(self): return np.asarray(self.det.mapping, dtype='i4') if self.det != NULL else None

