#ifndef EMC_H
#define EMC_H

#include "detector.h"
#include "dataset.h"
#include "quat.h"
#include "params.h"
#include "iterate.h"
#include "interp.h"

struct detector *det ;
struct rotation *quat ;
struct dataset *frames ;
struct iterate *iter ;
struct params *param ;

struct max_data {
	// Flags
	int refinement ; // Whether refinement data or global search
	int within_openmp ; // Whether this struct is local to a thread or not
	
	// Private to OpenMP thread only
	double *model, *weight ; // Thread-local copies of iterate
	double **all_views ; // View (W_rt) for each detector
	double *psum_r ; // S_r = \sum_d P_dr \phi_d
	double *psum_d ; // S_d = \sum_r P_dr u_r
	
	// Common among all threads only
	double *max_exp ; // max_exp[d] = max_r log(R_dr)
	double *p_norm ; // P_dr normalization, \sum_r R_dr
	double **u ; // u[detn][r] = -sum_t W_rt for every detn
	int *offset_prob ; // offset_prob[d*num_threads + thread] = num_prob offset for each OpenMP thread
	
	// Both
	double *max_exp_p ; // For priv, thread-local max_r. For common, process-local max_r
	double *info, *likelihood ; // Mutual information and log-likelihood for each d
	int *rmax ; // Most likely orientation for each d
	double *quat_norm ; // quat_norm[d, mode] = \sum_r P_dr for r in mode (if num_modes > 1)
	double **prob ; // prob[d][r] = P_dr
	int *num_prob ; // num_prob[d] = Number of non-zero prob[d][r] entries for each d
	int **place_prob ; // place_prob[d][r] = Position of non-zero prob[d][r]
	
	// Background-scaling update (private only)
	uint8_t **mask ; // Flag mask (M_t) used to optimize update
	double **G_old, **G_new, **G_latest ; // Gradients
	double **W_old, **W_new, **W_latest ; // Tomograms
	double *scale_old, *scale_new, *scale_latest ; // Scale factors
} ;

// setup_emc.c
int setup(char*, int) ;
void free_mem(void) ;

// max_emc.c
double maximize(void) ;

// output_emc.c
void write_log_file_header(int) ;
void update_log_file(double, double, double) ;
void save_initial_iterate(void) ;
void save_models(void) ;
void save_metrics(struct max_data*) ;
void save_prob(struct max_data*) ;

// interp function pointers
void (*slice_gen)(double*, double, double*, double*, long, struct detector*) ;
void (*slice_merge)(double*, double*, double*, double*, long, struct detector*) ;

#endif
