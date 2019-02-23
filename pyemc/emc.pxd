from libc.stdint cimport uint8_t
cimport decl

cdef extern from "../src/emc.h" nogil:
    decl.detector *det
    decl.rotation *quat
    decl.dataset *frames
    decl.iterate *iter
    decl.params *param
    
    struct max_data:
        # Flags
        int refinement # Whether refinement data or global search
        int within_openmp # Whether this struct is local to a thread or not
        
        # Private to OpenMP thread only
        double *model
        double *weight # Thread-local copies of iterate
        double **all_views # View (W_rt) for each detector
        double *psum_r # S[r] = \sum_d P_dr \phi_d
        double *psum_d # S[d] = \sum_r P_dr u_r
        
        # Common among all threads only
        double *max_exp # max_exp[d] = max_r log(R_dr)
        double *p_norm # P_dr normalization, \sum_r R_dr
        double **u # u[detn][r] = -sum_t W_rt for every detn
        int *offset_prob # offset_prob[d*num_threads + thread] = num_prob offset for each OpenMP thread
        
        # Both
        double *max_exp_p # For priv, thread-local max_r. For common, process-local max_r
        double *info # Mutual information for each d
        double *likelihood # Log-likelihood for each d
        int *rmax # Most likely orientation for each d
        double *quat_norm # quat_norm[d, mode] = \sum_r P_dr for r in mode (if num_modes > 1)
        double **prob # prob[d][r] = P_dr
        int *num_prob # num_prob[d] = Number of non-zero prob[d][r] entries for each d
        int **place_prob # place_prob[d][r] = Position of non-zero prob[d][r]
        
        # Background-scaling update (private only)
        uint8_t **mask # Flag mask (M[detn][t]) used to optimize update
        double **G_old
        double **G_new
        double **G_latest # Gradients
        double **W_old
        double **W_new
        double **W_latest # Tomograms
        double *scale_old
        double *scale_new
        double *scale_latest # Scale factors

    # setup_emc.c
    int setup(char*, int)
    void free_mem()

    # max_emc.c
    double maximize()

    # recon_emc.c
    int parse_arguments(int, char**, int*, int*, char*)
    void emc()
    void update_model(double)

    # output_emc.c
    void write_log_file_header(int)
    void update_log_file(double, double)
    void save_initial_iterate()
    void save_models()
    void save_metrics(max_data*)
    void save_prob(max_data*)
