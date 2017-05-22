#include "iterate.h"

int parse_scale(char *fname, struct dataset *frames, struct iterate *iter) {
	int flag = 0 ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		if (!rank)
			fprintf(stderr, "Using uniform scale factors\n") ;
	}
	else {
		if (!rank)
			fprintf(stderr, "Using scale factors from %s\n", fname) ;
		flag = 1 ;
		int d ;
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			fscanf(fp, "%lf", &iter->scale[d]) ;
		fclose(fp) ;
	}
	
	return flag ;
}

void calc_scale(struct dataset *frames, struct detector *det, char* print_fname, struct iterate *iter) {
	long d_counter = 0, ones_counter, multi_counter ;
	int d, t ;
	struct dataset *curr ;
	curr = frames ;
	
	iter->scale = calloc(frames->tot_num_data, sizeof(double)) ;
	frames->count = calloc(frames->tot_num_data, sizeof(int)) ;
	
	while (curr != NULL) {
		ones_counter = 0 ;
		multi_counter = 0 ;
		for (d = 0 ; d < curr->num_data ; ++d) {
			iter->scale[d_counter + d] = 1. ;
			for (t = 0 ; t < curr->ones[d] ; ++t)
			if (det->mask[curr->place_ones[ones_counter + t]] < 1)
				frames->count[d_counter + d]++ ;
			
			for (t = 0 ; t < curr->multi[d] ; ++t)
			if (det->mask[curr->place_multi[multi_counter + t]] < 1)
				frames->count[d_counter + d] += curr->count_multi[multi_counter + t] ;
			
			ones_counter += curr->ones[d] ;
			multi_counter += curr->multi[d] ;
		}
		
		d_counter += curr->num_data ;
		curr = curr->next ;
	}
	
	if (print_fname != NULL) {
		FILE *fp = fopen(print_fname, "w") ;
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			fprintf(fp, "%.6e\n", iter->scale[d]) ;
		fclose(fp) ;
	}
}

void normalize_scale(struct dataset *frames, struct iterate *iter) {
	double mean_scale = 0. ;
	long d, x, vol = iter->size*iter->size*iter->size ;
	
	for (d = 0 ; d < frames->tot_num_data ; ++d)
		mean_scale += iter->scale[d] ;
	mean_scale /= frames->tot_num_data ;
	for (x = 0 ; x < vol ; ++x)
		iter->model1[x] *= mean_scale ;
	for (d = 0 ; d < frames->tot_num_data ; ++d)
		iter->scale[d] /= mean_scale ;
}

void parse_input(char *fname, double mean, char *print_fname, struct iterate *iter) {
	long vol = iter->size * iter->size * iter->size ;
	iter->model1 = malloc(vol * sizeof(double)) ;
	iter->model2 = malloc(vol * sizeof(double)) ;
	iter->inter_weight = malloc(vol * sizeof(double)) ;
	
	if (rank == 0) {
		FILE *fp = fopen(fname, "r") ;
		if (fp == NULL) {
			if (!rank)
				fprintf(stderr, "Random start\n") ;
			
			long x ;
			struct timeval t ;
			const gsl_rng_type *T ;
			gsl_rng_env_setup() ;
			T = gsl_rng_default ;
			gsl_rng *rng = gsl_rng_alloc(T) ;
			gettimeofday(&t, NULL) ;
			gsl_rng_set(rng, t.tv_sec + t.tv_usec) ;
			
			for (x = 0 ; x < vol ; ++x)
				iter->model1[x] = gsl_rng_uniform(rng) * mean ;
			
			gsl_rng_free(rng) ;
		}
		else {
			if (!rank)
				fprintf(stderr, "Starting from %s\n", fname) ;
			
			fread(iter->model1, sizeof(double), vol, fp) ;
			fclose(fp) ;
		}
	}
	
	if (print_fname != NULL) {
		FILE *fp = fopen(print_fname, "wb") ;
		fwrite(iter->model1, sizeof(double), vol, fp) ;
		fclose(fp) ;
	}
}

void free_iterate(int scale_flag, struct iterate *iter) {
	free(iter->model1) ;
	free(iter->model2) ;
	free(iter->inter_weight) ;
	if (scale_flag)
		free(iter->scale) ;
}
