import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport emc
from quat cimport rotation

cdef class rotation:
	def __init__(self):
		self.rot = <emc.rotation*> PyMem_Malloc(sizeof(emc.rotation))
		self.rot.icosahedral_flag = 0
		self.rot.quat = NULL

	def generate_quaternion(self, config_fname, config_section='emc'):
		cdef char* c_config_fname = config_fname
		cdef char* c_config_section = config_section
		emc.generate_quaternion(c_config_fname, c_config_section, self.rot)
	
	def quat_gen(self, int num_div):
		emc.quat_gen(num_div, self.rot)

	def parse_quat(self, fname):
		cdef char* c_fname = fname
		emc.parse_quat(c_fname, self.rot)

	def divide_quat(self, int rank, int num_proc):
		emc.divide_quat(rank, num_proc, self.rot)

	def free_quat(self):
		emc.free_quat(self.rot)

	@property
	def num_rot(self): return self.rot.num_rot
	@property
	def num_rot_p(self): return self.rot.num_rot_p
	@property
	def quat(self): return np.asarray(<double[:self.num_rot*5]> self.rot.quat).reshape(-1,5)
	@property
	def icosahedral_flag(self): return bool(self.rot.icosahedral_flag)
