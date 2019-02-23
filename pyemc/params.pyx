cimport decl
from libc.stdlib cimport malloc, free
from params cimport params

cdef class params:
    def __init__(self, allocate=True):
        if allocate:
            self._alloc()
        else:
            self.param = NULL

    def _alloc(self):
        self.param = <decl.params*> malloc(sizeof(decl.params))
        self.param.rank = 0
        self.param.num_proc = 1
        self.param.known_scale = 0
        self.param.start_iter = 1
        self.param.beta_period = 100
        self.param.beta_jump = 1.
        self.param.need_scaling = 0
        self.param.alpha = 0.
        self.param.beta = 1.
        self.param.beta_start = 1.
        self.param.sigmasq = 0.
        self.param.log_fname[:7] = "EMC.log"
        self.param.output_folder[:5] = "data/"
        self.param.modes = 0
        self.param.rot_per_mode = 0

    def generate_params(self, config_fname):
        cdef char* c_config_fname = config_fname
        decl.generate_params(c_config_fname, self.param)

    def generate_output_dirs(self):
        decl.generate_output_dirs(self.param)

    def free_params(self):
        free(self.param)
        self.param = NULL

    @property
    def rank(self): return self.param.rank if self.param != NULL else None
    @rank.setter
    def rank(self, val): self.param.rank = val

    @property
    def num_proc(self): return self.param.num_proc if self.param != NULL else None
    @num_proc.setter
    def num_proc(self, val): self.param.num_proc = val

    @property
    def iteration(self): return self.param.iteration if self.param != NULL else None
    @iteration.setter
    def iteration(self, val): self.param.iteration = val

    @property
    def current_iter(self): return self.param.current_iter if self.param != NULL else None
    @current_iter.setter
    def current_iter(self, val): self.param.current_iter = val

    @property
    def start_iter(self): return self.param.start_iter if self.param != NULL else None
    @start_iter.setter
    def start_iter(self, val): self.param.start_iter = val

    @property
    def num_iter(self): return self.param.num_iter if self.param != NULL else None
    @num_iter.setter
    def num_iter(self, val): self.param.num_iter = val

    # Properties related to configuration file
    @property
    def output_folder(self): return <bytes>self.param.output_folder if self.param != NULL else None
    @property
    def log_fname(self): return <bytes>self.param.log_fname if self.param != NULL else None
    @property
    def beta_period(self): return self.param.beta_period if self.param != NULL else None
    @property
    def need_scaling(self): return self.param.need_scaling if self.param != NULL else None
    @property
    def known_scale(self): return self.param.known_scale if self.param != NULL else None
    @property
    def alpha(self): return self.param.alpha if self.param != NULL else None
    @property
    def beta(self): return self.param.beta if self.param != NULL else None
    @property
    def beta_start(self): return self.param.beta_start if self.param != NULL else None
    @property
    def beta_jump(self): return self.param.beta_jump if self.param != NULL else None
    @property
    def modes(self): return self.param.modes if self.param != NULL else None
    @property
    def rot_per_mode(self): return self.param.rot_per_mode if self.param != NULL else None
    @property
    def sigmasq(self): return self.param.sigmasq if self.param != NULL else None
    
