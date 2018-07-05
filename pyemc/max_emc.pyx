import os
import sys
import numpy as np
from contextlib import contextmanager

from libc.stdlib cimport malloc, free
from libc.float cimport DBL_MAX
from posix.time cimport gettimeofday 
cimport numpy as np

cimport decl
cimport max_emc

# Globals that need to be filled up by setup()
cdef extern from "../src/emc.h":
	decl.detector *det ;
	decl.rotation *quat ;
	decl.dataset *frames ;
	decl.dataset *merge_frames ;
	decl.iterate *iter ;
	decl.params *param ;
	
	void setup(char*, int)

@contextmanager
def _silence_stderr(to=os.devnull):
	fdes = sys.stderr.fileno()
	
	def _redirect_to(to):
		sys.stderr.close()
		os.dup2(to.fileno(), fdes)
		sys.stderr = os.fdopen(fdes, 'w')
	
	with os.fdopen(os.dup(fdes), 'w') as old_stderr:
		with open(to, 'w') as fptr:
			_redirect_to(fptr)
		try:
			yield
		finally:
			_redirect_to(old_stderr)

cdef class py_max_data:
	def __init__(self, refinement=0, within_openmp=0):
		self.data = <max_data*>malloc(sizeof(max_data))
		
		self.data.within_openmp = int(within_openmp)
		self.data.refinement = int(refinement)
		# Common only
		self.data.max_exp = NULL
		self.data.u = NULL
		self.data.probab = NULL
		# OpenMP Private only
		self.data.model = NULL
		self.data.weight = NULL
		self.data.scale = NULL
		self.data.all_views = NULL
		# Both
		self.data.max_exp_p = NULL
		self.data.p_sum = NULL
		self.data.info = NULL
		self.data.likelihood = NULL
		self.data.rmax = NULL
		self.data.quat_norm = NULL

	@property
	def within_openmp(self):
		return self.data.within_openmp
	@within_openmp.setter
	def within_openmp(self, value):
		self.data.within_openmp = value
	@property
	def max_exp(self):
		return np.asarray(<double[:frames.tot_num_data]> self.data.max_exp) if self.data.max_exp != NULL else None
	@property
	def u(self):
		return np.asarray(<double[:quat.num_rot_p]> self.data.u) if self.data.u != NULL else None
	@property
	def probab(self):
		if self.data.probab == NULL:
			return None
		else:
			retval = np.empty((quat.num_rot_p, frames.tot_num_data))
			for i in range(quat.num_rot_p):
				retval[i] = np.asarray(<double[:frames.tot_num_data]> self.data.probab[i])
			return retval
	@property
	def model(self):
		return np.asarray(<double[:iter.size**3]> self.data.model).reshape(3*(iter.size,)) if self.data.model != NULL else None
	@property
	def weight(self):
		return np.asarray(<double[:iter.size**3]> self.data.weight).reshape(3*(iter.size,)) if self.data.weight != NULL else None
	@property
	def scale(self):
		return np.asarray(<double[:frames.tot_num_data]> self.data.scale) if self.data.scale != NULL else None
	@property
	def all_views(self):
		if self.data.all_views == NULL:
			return None
		else:
			retval = []
			for i in range(det.num_det):
				retval.append(<double[:det[i].num_pix]> self.data.all_views[i])
			return retval
	@property
	def max_exp_p(self):
		return np.asarray(<double[:frames.tot_num_data]> self.data.max_exp_p) if self.data.max_exp_p != NULL else None
	@property
	def p_sum(self):
		if self.within_openmp == 0:
			return np.asarray(<double[:frames.tot_num_data]> self.data.p_sum) if self.data.p_sum != NULL else None
		else:
			return np.asarray(<double[:det.num_det]> self.data.p_sum) if self.data.p_sum != NULL else None
	@property
	def info(self):
		return np.asarray(<double[:frames.tot_num_data]> self.data.info) if self.data.info != NULL else None
	@property
	def likelihood(self):
		return np.asarray(<double[:frames.tot_num_data]> self.data.likelihood) if self.data.likelihood != NULL else None
	@property
	def rmax(self):
		return np.asarray(<int[:frames.tot_num_data]> self.data.rmax) if self.data.rmax != NULL else None
	@property
	def quat_norm(self):
		return np.asarray(<double[:param.modes]> self.data.quat_norm) if self.data.quat_norm != NULL else None

cdef class py_maximize:
	def __init__(self, config_fname, quiet_setup=True):
		gettimeofday(&tm1, NULL)
		if config_fname is not None:
			if quiet_setup:
				with _silence_stderr(): setup(config_fname, 0)
			else:
				setup(config_fname, 0)

	def allocate_memory(self, py_max_data data):
		max_emc.allocate_memory(data.data)

	def calculate_rescale(self, py_max_data common_data):
		iter.rescale = max_emc.calculate_rescale(common_data.data)
		return iter.rescale

	def calculate_prob(self, int r, py_max_data priv_data, py_max_data common_data):
		max_emc.calculate_prob(r, priv_data.data, common_data.data)

	def normalize_prob(self, py_max_data priv_data, py_max_data common_data):
		max_emc.normalize_prob(priv_data.data, common_data.data)

	def update_tomogram(self, int r, py_max_data priv_data, py_max_data common_data):
		max_emc.update_tomogram(r, priv_data.data, common_data.data)

	def merge_tomogram(self, int r, py_max_data priv_data):
		max_emc.merge_tomogram(r, priv_data.data)

	def combine_information_omp(self, py_max_data priv_data, py_max_data common_data):
		max_emc.combine_information_omp(priv_data.data, common_data.data)

	def combine_information_mpi(self, py_max_data common_data):
		avg_likelihood = max_emc.combine_information_mpi(common_data.data)
		return avg_likelihood

	def save_output(self, py_max_data common_data):
		max_emc.save_output(common_data.data)

	def free_memory(self, py_max_data data):
		max_emc.free_memory(data.data)
