#ifndef MAXIMIZE_H
#define MAXIMIZE_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include "iterate.h"

struct max_data {
	// Iterate
	struct iterate *iter ;
	
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
} ;

double maximize(struct max_data*) ;
void free_max_data(struct max_data*) ;

// Model function pointers
extern void (*slice_gen)(double*, int, double*, struct detector*, struct model*) ;
extern void (*slice_merge)(double*, int, double*, struct detector*, struct model*) ;
extern void slice_gen3d(double*, int, double*, struct detector*, struct model*) ;
extern void slice_merge3d(double*, int, double*, struct detector*, struct model*) ;
extern void slice_gen2d(double*, int, double*, struct detector*, struct model*) ;
extern void slice_merge2d(double*, int, double*, struct detector*, struct model*) ;


#endif // MAXIMIZE_H
