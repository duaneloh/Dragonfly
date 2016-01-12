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
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>

#define PROB_MIN 0.000001

int rank, num_proc, iteration, num_iter, start_iter, num_rot_p, num_rot_shift ;
int size, center, num_rot, num_pix, rel_num_pix ;

double info, rms_change, alpha, beta ;
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

struct dataset *frames ;
int tot_num_data ;
double tot_mean_count ;

uint8_t *mask ;

char output_folder[999], log_fname[999] ;

// setup.c
int setup(char*, int) ;
void free_mem() ;

// max.c
double maximize() ;

// interp.c
void slice_gen(double*, double, double*, double*, double*) ;
void slice_merge(double*, double*, double*, double*, double*) ;

#endif
