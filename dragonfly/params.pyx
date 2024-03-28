import os.path as op
from configparser import ConfigParser

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcpy
from . cimport params as c_params
from .params cimport EMCParams, params

cdef class EMCParams:
    def __init__(self):
        self.par = <c_params.params*> calloc(1, sizeof(c_params.params))

        self.par.rank = 0
        self.par.num_proc = 1
        self.par.start_iter = 1
        self.par.known_scale = 0

        self.par.log_fname = <char*> malloc(1024)
        strcpy(self.par.log_fname, b'EMC.log')
        self.par.output_folder= <char*> malloc(1024)
        strcpy(self.par.output_folder , b'data/')
        self.par.rtype = c_params.RECON3D

        self.par.need_scaling = 0
        self.par.alpha = 0.
        self.par.beta_factor = -1.
        self.par.radius = 0.
        self.par.num_modes = 1
        self.par.nonrot_modes = 0
        self.par.rot_per_mode = 0

        self.par.beta_jump = 1.
        self.par.beta_period = 100
        self.par.radius_jump = 0.
        self.par.radius_period = 100

        self.par.oversampling = 10.
        self.par.sigmasq = 0.
        self.par.friedel_sym = 0
        self.par.save_prob = 0
        self.par.update_scale = 1
        self.par.refine = 0
        self.par.coarse_div = 0
        self.par.fine_div = 0

    def free(self):
        if self.par == NULL:
            return

        if self.par.log_fname != NULL:
            free(self.par.log_fname)
        if self.par.output_folder != NULL:
            free(self.par.output_folder)
        free(self.par)
        self.par = NULL

    def from_config(self, config_fname, section_name='emc'):
        config_folder = op.dirname(config_fname).encode()
        config = ConfigParser()
        config.read(config_fname)

        strcpy(self.par.output_folder, op.join(config_folder, config.get(section_name, 'output_folder', fallback='data/').encode()))
        strcpy(self.par.log_fname, op.join(config_folder, config.get(section_name, 'log_file', fallback='EMC.log').encode()))
        rtype = config.get(section_name, 'recon_type', fallback='3d')
        if rtype == '3d':
            self.par.rtype = c_params.RECON3D
        elif rtype == '2d':
            self.par.rtype = c_params.RECON2D
        elif rtype == 'rz':
            self.par.rtype = c_params.RECONRZ
        else:
            raise ValueError('Unknown recon_type %s'%rtype)
        
        self.par.verbosity = config.getint(section_name, 'verbosity', fallback=9)
        self.par.need_scaling = config.getint(section_name, 'need_scaling', fallback=0)
        self.par.alpha = config.getfloat(section_name, 'alpha', fallback=0.)
        self.par.beta_factor = config.getfloat(section_name, 'beta_factor', fallback=1.)
        self.par.radius = config.getfloat(section_name, 'radius', fallback=0.)
        self.par.num_modes = config.getint(section_name, 'num_modes', fallback=1)
        self.par.nonrot_modes = config.getint(section_name, 'num_nonrot_modes', fallback=0)
        self.par.rot_per_mode = config.getint(section_name, 'num_rot', fallback=0)

        beta_schedule = tuple(config.get(section_name, 'beta_schedule', fallback='1. 100').split())
        self.par.beta_jump = float(beta_schedule[0])
        self.par.beta_period = int(beta_schedule[1])
        radius_schedule= tuple(config.get(section_name, 'radius_schedule', fallback='0. 100').split())
        self.par.radius_jump = float(radius_schedule[0])
        self.par.radius_period = int(radius_schedule[1])

        self.par.oversampling = config.getfloat(section_name, 'oversampling', fallback=10.)
        self.par.sigmasq = config.getfloat(section_name, 'gaussian_sigma', fallback=0.)**2
        self.par.friedel_sym = config.getint(section_name, 'friedel_sym', fallback=0)
        self.par.save_prob = config.getint(section_name, 'save_prob', fallback=0)
        self.par.update_scale = config.getint(section_name, 'update_scale', fallback=1)
        num_divs = config.get(section_name, 'num_div', fallback='0').split()
        if len(num_divs) > 1:
            self.par.refine = 1
            self.par.fine_div = int(num_divs[0])
            self.par.coarse_div = int(num_divs[1])
            print("Doing refinement from num_div = %d -> %d\n" % (self.par.coarse_div, self.par.fine_div))

    @property
    def rank(self): return self.par.rank
    @rank.setter
    def rank(self, int val): self.par.rank = val
    @property
    def num_proc(self): return self.par.num_proc
    @num_proc.setter
    def num_proc(self, int val): self.par.num_proc = val
    @property
    def iteration(self): return self.par.iteration
    @iteration.setter
    def iteration(self, int val): self.par.iteration = val
    @property
    def start_iter(self): return self.par.start_iter
    @start_iter.setter
    def start_iter(self, int val): self.par.start_iter = val
    @property
    def num_iter(self): return self.par.num_iter
    @num_iter.setter
    def num_iter(self, int val): self.par.num_iter = val
    @property
    def beta_period(self): return self.par.beta_period
    @beta_period.setter
    def beta_period(self, int val): self.par.beta_period = val
    @property
    def need_scaling(self): return self.par.need_scaling
    @need_scaling.setter
    def need_scaling(self, int val): self.par.need_scaling = val
    @property
    def known_scale(self): return self.par.known_scale
    @known_scale.setter
    def known_scale(self, int val): self.par.known_scale = val
    @property
    def update_scale(self): return self.par.update_scale
    @update_scale.setter
    def update_scale(self, int val): self.par.update_scale = val
    @property
    def save_prob(self): return self.par.save_prob
    @save_prob.setter
    def save_prob(self, int val): self.par.save_prob = val
    @property
    def verbosity(self): return self.par.verbosity
    @verbosity.setter
    def verbosity(self, int val): self.par.verbosity = val
    @property
    def friedel_sym(self): return self.par.friedel_sym
    @friedel_sym.setter
    def friedel_sym(self, int val): self.par.friedel_sym = val
    @property
    def refine(self): return self.par.refine
    @refine.setter
    def refine(self, int val): self.par.refine = val
    @property
    def coarse_div(self): return self.par.coarse_div
    @coarse_div.setter
    def coarse_div(self, int val): self.par.coarse_div = val
    @property
    def fine_div(self): return self.par.fine_div
    @fine_div.setter
    def fine_div(self, int val): self.par.fine_div = val
    @property
    def radius_period(self): return self.par.radius_period
    @radius_period.setter
    def radius_period(self, int val): self.par.radius_period = val
    @property
    def num_modes(self): return self.par.num_modes
    @num_modes.setter
    def num_modes(self, int val): self.par.num_modes = val
    @property
    def rot_per_mode(self): return self.par.rot_per_mode
    @rot_per_mode.setter
    def rot_per_mode(self, int val): self.par.rot_per_mode = val
    @property
    def nonrot_modes(self): return self.par.nonrot_modes
    @nonrot_modes.setter
    def nonrot_modes(self, int val): self.par.nonrot_modes = val

    @property
    def rtype(self): return ['RECON3D', 'RECON2D', 'RECONRZ'][self.par.rtype]
    @rtype.setter
    def rtype(self, val): 
        if val == 'RECON3D':
            self.par.rtype = RECON3D
        elif val == 'RECON2D':
            self.par.rtype = RECON2D
        elif val == 'RECONRZ':
            self.par.rtype = RECONRZ
        else:
            raise ValueError('Unknown recon_type %s'%val)
            
    @property
    def alpha(self): return self.par.alpha
    @alpha.setter
    def alpha(self, double val): self.par.alpha = val
    @property
    def beta_jump(self): return self.par.beta_jump
    @beta_jump.setter
    def beta_jump(self, double val): self.par.beta_jump = val
    @property
    def beta_factor(self): return self.par.beta_factor
    @beta_factor.setter
    def beta_factor(self, double val): self.par.beta_factor = val
    @property
    def radius(self): return self.par.radius
    @radius.setter
    def radius(self, double val): self.par.radius = val
    @property
    def radius_jump(self): return self.par.radius_jump
    @radius_jump.setter
    def radius_jump(self, double val): self.par.radius_jump = val
    @property
    def oversampling(self): return self.par.oversampling
    @oversampling.setter
    def oversampling(self, double val): self.par.oversampling = val
    @property
    def sigmasq(self): return self.par.sigmasq
    @sigmasq.setter
    def sigmasq(self, double val): self.par.sigmasq = val
    
    @property
    def log_fname(self): return (<bytes> self.par.log_fname).decode() if self.par.log_fname != NULL else None
    @log_fname.setter
    def log_fname(self, val):
        if self.par.log_fname != NULL:
            free(self.par.log_fname)
        self.par.log_fname = <char*> malloc(len(val) + 1)
        strcpy(self.par.log_fname, bytes(val, 'utf-8'))
    @property
    def output_folder(self): return (<bytes> self.par.output_folder).decode() if self.par.output_folder != NULL else None
    @output_folder.setter
    def output_folder(self, val):
        if self.par.output_folder != NULL:
            free(self.par.output_folder)
        self.par.output_folder = <char*> malloc(len(val) + 1)
        strcpy(self.par.output_folder, bytes(val, 'utf-8'))

