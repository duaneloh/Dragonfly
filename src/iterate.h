#ifndef ITERATE_H
#define ITERATE_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <gsl/gsl_rng.h>
#include "dataset.h"
#include "detector.h"

extern int rank ;

struct iterate {
	long size, center ;
	double *model1, *model2, *inter_weight ;
	double *scale ;
	
	double rescale, mutual_info, rms_change ;
} ;

int parse_scale(char*, struct dataset*, struct iterate*) ;
void calc_scale(struct dataset*, struct detector*, char*, struct iterate*) ;
void normalize_scale(struct dataset*, struct iterate*) ;
void parse_input(char*, double, char*, struct iterate*) ;
void free_iterate(int, struct iterate*) ;

#endif //ITERATE_H
