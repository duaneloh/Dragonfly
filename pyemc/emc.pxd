from libc.stdint cimport uint8_t
from libc.stdio cimport FILE
cimport numpy as np

cdef extern from 'numpy/arrayobject.h':
	void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from '../src/emc.h':
	int rank, num_proc
	char config_section[1024]
config_section[:] = ['\0']*1024

cdef extern from '../src/detector.h':
	struct detector:
		int num_pix, rel_num_pix
		double detd, ewald_rad
		double *pixels
		uint8_t *mask
		
		# Only relevant for first detector in list
		int num_det, num_dfiles
		int mapping[1024]

	double generate_detectors(FILE*, detector**, int)                          #####################
	double parse_detector(char*, detector*, int)                               # ==========> Wrapped
	double parse_detector_list(char*, detector**, int)                         #####################
	void free_detector(detector*)                                              # ==========> Wrapped

cdef extern from '../src/dataset.h':
	struct dataset:
		int num_data, num_pix, num_data_prev
		# Data set type (0=sparse, 1=dense integer, 2=dense double)
		int type
		double mean_count
		char filename[1024]
		
		# Sparse dataset
		long ones_total, multi_total
		int *ones
		int *multi
		int *place_ones
		int *place_multi
		int *count_multi
		long *ones_accum
		long *multi_accum
		
		# Dense dataset
		double *frames
		int *int_frames
		
		# Pointer to next dataset
		dataset *next
		
		# Need only be defined for head dataset
		int tot_num_data, num_blacklist
		double tot_mean_count
		int *count
		double *sum_fact
		uint8_t *blacklist

	int generate_data(FILE*, char*, dataset*, detector*)                       #####################
	void calc_sum_fact(detector*, dataset*)                                    # ==========> Wrapped
	int parse_dataset(char*, detector*, dataset*)                              # ==========> Wrapped
	int parse_data(char*, detector*, dataset*)                                 # ==========> Wrapped
	void make_blacklist(char*, int, dataset*)                                  # ==========> Wrapped
	void free_data(int, dataset*)                                              # ==========> Wrapped

cdef extern from '../src/quat.h':
	struct rotation:
		int num_rot, num_rot_p
		double *quat
		int icosahedral_flag

	int generate_quaternion(FILE*, rotation*)                                  # ==========> Wrapped
	int quat_gen(int, rotation*)                                               # ==========> Wrapped
	int parse_quat(char*, rotation*)                                           # ==========> Wrapped
	void divide_quat(int, int, rotation*)                                      # ==========> Wrapped
	void free_quat(rotation*)                                                  # ==========> Wrapped

cdef extern from '../src/iterate.h':
	struct iterate:
		long size, center
		double *model1
		double *model2
		double *inter_weight
		double *scale
		
		double rescale, mutual_info, rms_change

	void generate_size(double, iterate*)                                       # ==========> Wrapped
	int parse_scale(char*, dataset*, iterate*)                                 # ==========> Wrapped
	void calc_scale(dataset*, detector*, char*, iterate*)                      # ==========> Wrapped
	void normalize_scale(dataset*, iterate*)                                   # ==========> Wrapped
	void parse_input(char*, double, char*, iterate*)                           # ==========> Wrapped
	void free_iterate(int, iterate*)                                           # ==========> Wrapped

cdef extern from '../src/interp.h':
	void make_rot_quat(double*, double[3][3])                                  # ==========> Wrapped
	void slice_gen(double*, double, double*, double*, long, detector*)         # ==========> Wrapped
	void slice_merge(double*, double*, double*, double*, long, detector*)      # ==========> Wrapped
	void rotate_model(double[3][3], double*, int, double*)                     # ==========> Wrapped
	void symmetrize_friedel(double*, int)                                      # ==========> Wrapped

cdef extern from '../src/params.h':
	struct params:
		int iteration, current_iter, start_iter, num_iter ;
		char output_folder[1024]
		char log_fname[1024] ;
		
		# Algorithm parameters
		int beta_period, need_scaling, known_scale ;
		double alpha, beta, beta_jump ;
		
		# Gaussian EMC parameter
		double sigmasq ;

