import numpy as np
cimport numpy as np
cimport emc
from quat cimport rotation

cdef class rotation:
	def __init__(self):
		pass

	@property
	def num_rot(self): return self.rot.num_rot
	@property
	def num_rot_p(self): return self.rot.num_rot_p
	@property
	def quat(self): return np.asarray(<double[:self.num_rot*5]> self.rot.quat).reshape(-1,5)
	@property
	def icosahedral_flag(self): return self.rot.icosahedral_flag
