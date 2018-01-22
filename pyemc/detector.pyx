import numpy as np
cimport numpy as np
cimport emc 
cimport openmp
from detector cimport detector

cdef class detector:
	def __init__(self):
		pass

	def parse_detector(self, fname, norm_flag=True):
		cdef char* c_fname = fname
		emc.parse_detector(c_fname, self.det, int(norm_flag))

	def free_detector(self):
		emc.free_detector(self.det)

	def generate_size(double qmax, np.ndarray[long] size, np.ndarray[long] center):
		emc.generate_size(qmax, &size[0], &center[0])

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



