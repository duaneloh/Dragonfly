import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport emc 
cimport openmp
from detector cimport detector

cdef class detector:
	def __init__(self):
		self.det = <emc.detector*> PyMem_Malloc(sizeof(emc.detector))
		self.det.num_det = 0

	def generate_detectors(self, config_fname, norm_flag=True, config_section='emc', rank=0, num_proc=1):
		if emc.config_section[0] == '\0':
			emc.config_section[:len(config_section)] = config_section
			emc.config_section[len(config_section)] = '\0'
		if emc.rank == 0: emc.rank = rank
		if emc.num_proc == 0: emc.num_proc = num_proc
		
		cdef char* c_config_fname = config_fname
		if self.det.num_det > 0:
			self.free_detector()
		emc.generate_detectors(c_config_fname, &self.det, int(norm_flag))

	def parse_detector_list(self, flist, norm_flag=True):
		cdef char* c_flist = flist
		if self.det.num_det > 0:
			self.free_detector()
		emc.parse_detector_list(flist, &self.det, int(norm_flag))

	def parse_detector(self, fname, norm_flag=True):
		cdef char* c_fname = fname
		if self.det.num_det > 0:
			emc.free_detector(self.det)
		emc.parse_detector(c_fname, self.det, int(norm_flag))
		self.det.num_det = 1

	def free_detector(self):
		emc.free_detector(self.det)
		PyMem_Free(self.det)

	@property
	def num_pix(self): return self.det.num_pix
	@property
	def rel_num_pix(self): return self.det.rel_num_pix
	@property
	def detd(self): return self.det.detd
	@property
	def ewald_rad(self): return self.det.ewald_rad
	@property
	def pixels(self): return np.asarray(<double[:4*self.num_pix]>self.det.pixels).reshape(-1,4)
	@property
	def mask(self): return np.asarray(<emc.uint8_t[:self.num_pix]>self.det.mask)

	# Only relevant for first detector in list
	@property
	def num_det(self): return self.det.num_det
	@property
	def num_dfiles(self): return self.det.num_dfiles
	@property
	def mapping(self): return np.asarray(self.det.mapping, dtype='i4')



