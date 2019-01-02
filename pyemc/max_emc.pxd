from posix.time cimport timeval

cdef extern from "../src/max_emc.c" nogil:
	struct max_data:
		int refinement
		int within_openmp
		
		# Common only
		double *max_exp
		double *u
		double **probab
		
		# Private only
		double *model
		double *weight
		double *scale
		double **all_views
		
		# Both
		double *max_exp_p
		double *p_sum
		double *info
		double *likelihood
		int *rmax
		double *quat_norm

	timeval tm1
	timeval tm2

	cdef void allocate_memory(max_data*)
	cdef double calculate_rescale(max_data*)
	cdef void calculate_prob(int, max_data*, max_data*)
	cdef void normalize_prob(max_data*, max_data*)
	cdef void update_tomogram(int, max_data*, max_data*)
	cdef void merge_tomogram(int, max_data*)
	cdef void combine_information_omp(max_data*, max_data*)
	cdef double combine_information_mpi(max_data*)
	cdef void save_output(max_data*)
	cdef void free_memory(max_data*)

cdef class py_max_data:
	cdef max_data* data

cdef class py_maximize:
	cdef max_data* common_data
	cdef max_data* priv_data

