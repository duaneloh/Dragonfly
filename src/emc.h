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
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include "detector.h"
#include "dataset.h"
#include "interp.h"
#include "quat.h"

#define PROB_MIN 0.000001

struct detector *det ;
struct rotation *quat ;
struct dataset *frames, *merge_frames ;

int rank, num_proc, iteration, start_iter ;
int size, center, beta_period ;

double mutual_info, rms_change, alpha, beta, beta_jump ;
double *model1, *model2, *inter_weight ;

double *sum_fact ;
double *scale ;
int *count, need_scaling, known_scale ;
uint8_t *blacklist ;
int num_blacklist ;

char output_folder[999], log_fname[999] ;

// setup.c
int setup(char*, int) ;
void free_mem() ;

// max.c
double maximize() ;

#endif
