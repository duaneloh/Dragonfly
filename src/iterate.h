#ifndef ITERATE_H
#define ITERATE_H

#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <sys/time.h>
#include <gsl/gsl_rng.h>
#ifdef WITH_HDF5
#include <hdf5.h>
#endif // WITH_HDF5
#include "utils.h"
#include "dataset.h"
#include "detector.h"
#include "params.h"
#include "quat.h"

struct iterate {
	long size, center, vol ;
	int modes, tot_num_data ;
	double *model1, *model2, *inter_weight ;
	double *scale, *bgscale ;

	// For refinement
	int *quat_mapping, **rel_quat, *num_rel_quat ;
	double **rel_prob ;
	
	double *rescale, *mean_count ; // For each detector
	double mutual_info, rms_change ;
} ;

int iterate_from_config(char*, char*, int, double, struct params*, struct detector*, struct dataset*, struct iterate*) ;
void calculate_size(double, struct iterate*) ;
int parse_scale(char*, double*, struct iterate*) ;
void calc_scale(struct dataset*, struct detector*, struct iterate*) ;
void normalize_scale(struct dataset*, struct iterate*) ;
void parse_input(char*, double, int, int, struct iterate*) ;
int parse_rel_quat(char*, int, int, struct iterate*) ;
void calc_beta(struct dataset*, struct params*) ;
void free_iterate(struct iterate*) ;

#endif //ITERATE_H
