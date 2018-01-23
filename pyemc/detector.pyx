import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdio cimport FILE, fdopen
cimport emc 
cimport openmp
from detector cimport detector

cdef class detector:
	def __init__(self):
		self.det = <emc.detector*> PyMem_Malloc(sizeof(emc.detector))

	def generate_detectors(self, config_file, norm_flag=True, config_section='emc', rank=0, num_proc=1):
		if emc.config_section[0] == '\0':
			emc.config_section[:len(config_section)] = config_section
			emc.config_section[len(config_section)] = '\0'
		if emc.rank == 0: emc.rank = rank
		if emc.num_proc == 0: emc.num_proc = num_proc
		
		cdef FILE* config_fp = fdopen(config_file.fileno(), 'r')
		emc.generate_detectors(config_fp, &self.det, int(norm_flag))

	def parse_detector(self, fname, norm_flag=True):
		cdef char* c_fname = fname
		emc.parse_detector(c_fname, self.det, int(norm_flag))

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



