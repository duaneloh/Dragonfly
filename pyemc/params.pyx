cimport emc
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from params cimport params

cdef class params:
	def __init__(self):
		self.param = <emc.params*> PyMem_Malloc(sizeof(emc.params))
		self.param.rank = 0
		self.param.num_proc = 1
		self.param.known_scale = 0
		self.param.start_iter = 1
		self.param.beta_period = 100
		self.param.beta_jump = 1.
		self.param.need_scaling = 0
		self.param.alpha = 0.
		self.param.beta = 1.
		self.param.sigmasq = 0.
		self.param.log_fname[:7] = "EMC.log"
		self.param.output_folder[:5] = "data/"

	def generate_params(self, config_fname):
		cdef char* c_config_fname = config_fname
		emc.generate_params(c_config_fname, self.param)

	def generate_output_dirs(self):
		emc.generate_output_dirs(self.param)

	@property
	def rank(self): return self.param.rank
	@property
	def num_proc(self): return self.param.num_proc
	@property
	def output_folder(self): return str(self.param.output_folder)
	@property
	def log_fname(self): return str(self.param.log_fname)
	@property
	def iteration(self): return self.param.iteration
	@property
	def current_iter(self): return self.param.current_iter
	@property
	def start_iter(self): return self.param.start_iter
	@property
	def num_iter(self): return self.param.num_iter
	@property
	def beta_period(self): return self.param.beta_period
	@property
	def need_scaling(self): return self.param.need_scaling
	@property
	def known_scale(self): return self.param.known_scale
	@property
	def alpha(self): return self.param.alpha
	@property
	def beta(self): return self.param.beta
	@property
	def beta_jump(self): return self.param.beta_jump
	@property
	def sigmasq(self): return self.param.sigmasq
	
