#ifndef EMC_H
#define EMC_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <float.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>

#define PROB_MIN 0.000001

int rank, num_proc, iteration, num_iter, start_iter, num_rot_p ;
int size, center, num_rot, num_pix, rel_num_pix, beta_period ;

double mutual_info, rms_change, alpha, beta, beta_jump ;
double *det, *quat, *model1, *model2, *inter_weight ;

double *sum_fact ;
double *scale ;
int *count, need_scaling, known_scale ;
uint8_t *blacklist ;
int num_blacklist ;

struct dataset {
	int num_data, num_pix ;
	long ones_total, multi_total ;
	double mean_count ;
	int *ones, *multi, *place_ones, *place_multi, *count_multi ;
	struct dataset *next ;
	char filename[999] ;
} ;

struct dataset *frames, *merge_frames ;
int tot_num_data ;
double tot_mean_count ;

uint8_t *mask ;

char output_folder[999], log_fname[999] ;
int icosahedral_flag ;

// setup.c
int setup(char*, int) ;
void free_mem() ;

// max.c
double maximize() ;

// interp.c
void slice_gen(double*, double, double*, double*, double*) ;
void slice_merge(double*, double*, double*, double*, double*) ;
void symmetrize_icosahedral(double*, int) ;

// quat.c
int quat_gen(int, double**, int) ;

#endif
