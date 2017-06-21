#ifndef PARAMS_H
#define PARAMS_H

struct params {
	int iteration, current_iter, start_iter, num_iter ;
	char output_folder[1024], log_fname[1024] ;
	double rescale ;
	
	// Algorithm parameters
	int beta_period, need_scaling, known_scale ;
	double alpha, beta, beta_jump ;
} ;

#endif //PARAMS_H
