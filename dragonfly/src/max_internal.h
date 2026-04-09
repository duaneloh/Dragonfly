#ifndef MAX_INTERNAL_H
#define MAX_INTERNAL_H

#include "maximize.h"

#define PROB_MIN 1.e-6
#define PDIFF_THRESH 14.
#define MAX_EXP_START -1.e100

extern struct timeval tm1, tm2 ;

// Function pointers (defined in maximize.c)
extern void (*slice_gen)(double*, int, double*, struct detector*, struct model*) ;
extern void (*slice_merge)(double*, int, double*, double*, double*, long, struct detector*) ;

// maximize.c
void allocate_memory(struct max_data*) ;
void free_max_data(struct max_data*) ;
void print_max_time(char*, char*, int) ;

// probability.c
void calculate_rescale(struct max_data*) ;
void calculate_prob(int, struct max_data*, struct max_data*) ;
void normalize_prob(struct max_data*, struct max_data*) ;
void combine_information_omp(struct max_data*, struct max_data*) ;
double combine_information_mpi(struct max_data*) ;
int resparsify(double*, int*, int, double) ;

// tomogram.c
void update_tomogram(int, struct max_data*, struct max_data*) ;
void merge_tomogram(int, struct max_data*) ;
double calc_psum_r(int, struct max_data*, struct max_data*) ;
void update_tomogram_nobg(int, struct max_data*, struct max_data*) ;
void gradient_rt(int, struct max_data*, double**, double**) ;
void update_tomogram_bg(int, double, struct max_data*, struct max_data*) ;

// scaling.c
void update_scale(struct max_data*) ;
void gradient_d(struct max_data*, uint8_t*, double*, double*) ;
void update_scale_bg(struct max_data*) ;

#endif // MAX_INTERNAL_H
