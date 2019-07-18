from libc.stdint cimport uint8_t

cdef extern from '../src/detector.h' nogil:
    struct detector:
        int num_pix, rel_num_pix
        double detd, ewald_rad
        double *pixels
        uint8_t *mask
        
        # Only relevant for first detector in list
        int num_det, num_dfiles
        int mapping[1024]
        
        # Powder sum for EMC
        double *powder

    double generate_detectors(char*, char*, detector**, int)
    double parse_detector(char*, detector*, int)
    double parse_detector_list(char*, detector**, int)
    void free_detector(detector*)

cdef extern from '../src/dataset.h' nogil:
    struct dataset:
        int num_data, num_pix, num_data_prev
        # Data set type [0=sparse, 1=dense integer, 2=dense double]
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

    int generate_data(char*, char*, char*, detector*, dataset*)
    void calc_sum_fact(detector*, dataset*)
    int parse_dataset(char*, detector*, dataset*)
    int parse_dataset_list(char*, detector*, dataset*)
    void generate_blacklist(char*, dataset*)
    void make_blacklist(char*, int, dataset*)
    void calc_powder(detector*, dataset*)
    void free_data(int, dataset*)

cdef extern from '../src/quat.h' nogil:
    struct rotation:
        int num_rot, num_rot_p
        double *quat
        int icosahedral_flag

    int generate_quaternion(char*, char*, rotation*)
    int quat_gen(int, rotation*)
    int parse_quat(char*, int, rotation*)
    void divide_quat(int, int, int, rotation*)
    void voronoi_subset(rotation*, rotation*, int*)
    void free_quat(rotation*)

cdef extern from '../src/iterate.h' nogil:
    struct iterate:
        long size, center, vol
        int tot_num_data, modes
        double *model1
        double *model2
        double *inter_weight
        double *scale
        double *rescale
        int *quat_mapping
        int **rel_quat
        int *num_rel_quat
        
        double mutual_info, rms_change

    int generate_iterate(char*, char*, int, double, params*, detector*, dataset*, iterate*)
    void calculate_size(double, iterate*)
    int parse_scale(char*, iterate*)
    void calc_scale(dataset*, detector*, iterate*)
    void normalize_scale(dataset*, iterate*)
    void parse_input(char*, double, int, int, iterate*)
    int parse_rel_quat(char*, int, int, iterate*)
    void free_iterate(iterate*)

cdef extern from '../src/interp.h' nogil:
    void make_rot_quat(double*, double[3][3])
    void slice_gen3d(double*, double, double*, double*, long, detector*)
    void slice_merge3d(double*, double*, double*, double*, long, detector*)
    void slice_gen2d(double*, double, double*, double*, long, detector*)
    void slice_merge2d(double*, double*, double*, double*, long, detector*)
    void rotate_model(double[3][3], double*, int, double*)
    void symmetrize_friedel(double*, int)
    void symmetrize_icosahedral(double*, int)

cdef extern from '../src/params.h' nogil:
    struct params:
        int rank, num_proc
        int iteration, current_iter, start_iter, num_iter
        char output_folder[1024]
        char log_fname[1024]
        int recon_type
        
        # Algorithm parameters
        int beta_period, need_scaling, known_scale, update_scale
        double alpha, beta_start, beta, beta_jump
        int friedel_sym # Symmetrization for 2D recon
        int refine, coarse_div, fine_div # If doing refinement
        
        # Gaussian EMC parameter
        double sigmasq
        
        # Number of unconstrained modes
        int modes, rot_per_mode

    void generate_params(char*, params*)
    void generate_output_dirs(params*)
