#ifndef PARAMS_H
#define PARAMS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <sys/stat.h>
#include "utils.h"

#define RECON3D 42
#define RECON2D 43
#define RECONRZ 44

struct params {
	int rank, num_proc ;
	int iteration, current_iter, start_iter, num_iter ;
	char output_folder[1024], log_fname[1024] ;
	int recon_type, save_prob ;
	
	// Algorithm parameters
	int beta_period, need_scaling, known_scale, update_scale ;
	double alpha, beta_jump, beta_factor ;
	double *beta, *beta_start ;
	int friedel_sym ; // Symmetrization for 2D recon
	int axial_sym ; // N-fold symmetrization about Z-axis
	int refine, coarse_div, fine_div ; // If doing refinement

	// Radius refinement
	int radius_period ;
	double radius, radius_jump, oversampling ;
	
	// Gaussian EMC parameter
	double sigmasq ;

	// Number of unconstrained modes
	int modes, rot_per_mode, nonrot_modes ;
} ;

void params_from_config(char*, char*, struct params*) ;
void generate_output_dirs(struct params*) ;
void free_params(struct params*) ;

#endif //PARAMS_H
