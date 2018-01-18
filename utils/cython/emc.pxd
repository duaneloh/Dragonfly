from libc.stdint cimport uint8_t
from libc.stdio cimport FILE

cdef extern from 'common.h':
	int rank, num_proc
	char config_section[1024]

cdef extern from '../../src/detector.h':
	struct detector:
		int num_pix, rel_num_pix
		double detd, ewald_rad
		double *pixels
		uint8_t *mask
		
		# Only relevant for first detector in list
		int num_det, num_dfiles
		int mapping[1024]
	double generate_detectors(FILE*, detector**, int)
	void generate_size(double, long*, long*)
	double parse_detector(char*, detector*, int) # ==========> Wrapped
	double parse_detector_list(char*, detector**, int)
	void free_detector(detector*) # ==========> Wrapped


cdef extern from "../../src/interp.h":
	void make_rot_quat(double*, double[3][3])
	void slice_gen(double*, double, double*, double*, long, detector*) # ==========> Wrapped
	void slice_merge(double*, double*, double*, double*, long, detector*) # ==========> Wrapped
	void rotate_model(double[3][3], double*, int, double*)
	void symmetrize_friedel(double*, int) # ==========> Wrapped

