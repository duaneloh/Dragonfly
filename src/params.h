#ifndef PARAMS_H
#define PARAMS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <sys/stat.h>

#define RECON3D 42
#define RECON2D 43

struct params {
	int rank, num_proc ;
	int iteration, current_iter, start_iter, num_iter ;
	char output_folder[1024], log_fname[1024] ;
	int recon_type ;
	
	// Algorithm parameters
	int beta_period, need_scaling, known_scale ;
	double alpha, beta, beta_jump ;
	
	// Gaussian EMC parameter
	double sigmasq ;

	// Number of unconstrained modes
	int modes, rot_per_mode ;
} ;

void generate_params(char*, struct params*) ;
void generate_output_dirs(struct params*) ;

#endif //PARAMS_H
