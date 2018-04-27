#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <inttypes.h>
#include <limits.h>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <math.h>
#include <sys/time.h>
#include <float.h>
#include <stdint.h>
#include <libgen.h>
#include "../../src/detector.h"
#include "../../src/interp.h"

#define NUM_AVE 5000
#define FLUENCE 0
#define COUNTS 1

int testing_mode = 0 ;
int size, num_rot, scale_method ;
int **place_ones, **place_multi, *ones, *multi, **count_multi ;
double *intens, *likelihood, *quat_list, *scale_factors ;
struct detector *det ;

// Config file params
int num_data, do_gamma ;
double fluence, rescale, mean_count, background ;
char output_fname[1024], likelihood_fname[1024], scale_fname[1024] ;

void rescale_intens() ;
void allocate_data_memory() ;
double calc_dataset() ;
void write_dataset() ;
int setup(char*) ;
void free_mem() ;

int main(int argc, char *argv[]) {
	int c ;
	double actual_mean_count ;
	struct timeval t1, t2 ;
	gsl_rng_env_setup() ;
	
	char config_fname[1024] ;
	extern char *optarg ;
	extern int optind ;
	
	omp_set_num_threads(omp_get_max_threads()) ;
	strcpy(config_fname, "config.ini") ;
	while ((c = getopt(argc, argv, "c:t:Th")) != -1) {
		switch (c) {
			case 't':
				omp_set_num_threads(atoi(optarg)) ;
				break ;
			case 'c':
				strcpy(config_fname, optarg) ;
				break ;
			case 'T':
				testing_mode = 1 ;
				fprintf(stderr, "====== Testing mode (fixed seed) ======\n") ;
				break ;
			case 'h':
				fprintf(stderr, "Format: %s [-c config_fname] [-t num_threads] [-h]\n", argv[0]) ;
				return 1 ;
		}
	}
	fprintf(stderr, "Generating data with parameters from %s\n", config_fname) ;
	
	if (setup(config_fname))
		return 1 ;
	
	gettimeofday(&t1, NULL) ;
	
	rescale_intens() ;
	allocate_data_memory() ;
	actual_mean_count = calc_dataset() ;
	write_dataset() ;
	
	gettimeofday(&t2, NULL) ;
	fprintf(stderr, "Generated %d frames with %f photons/frame\n", num_data, actual_mean_count) ;
	fprintf(stderr, "Time taken = %f s\n", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;
	
	free_mem() ;
	
	return 0 ;
}

void rand_quat(double quat[4], gsl_rng *rng) {
	int i ;
	double qq ;
	
	do {
		qq = 0. ;
		for (i = 0 ; i < 4 ; ++i) {
			quat[i] = gsl_rng_uniform(rng) -.5 ;
			qq += quat[i] * quat[i] ;
		}
	}
	while (qq > .25) ;
	
	qq = sqrt(qq) ;
	for (i = 0 ; i < 4 ; ++i)
		quat[i] /= qq ;
}

void rescale_intens() {
	int x ;
	double rescale = 0., intens_ave = 0. ;
	const gsl_rng_type *T = gsl_rng_default ;
	gsl_rng *rng = gsl_rng_alloc(T) ;
	unsigned long *seeds = malloc(omp_get_max_threads() * sizeof(unsigned long)) ;
	
	if (testing_mode) {
		gsl_rng_set(rng, 0x5EED) ;
	}
	else {
		struct timeval tval ;
		gettimeofday(&tval, NULL) ;
		gsl_rng_set(rng, tval.tv_sec + tval.tv_usec) ;
	}
	for (x = 0 ; x < omp_get_max_threads() ; ++x)
		seeds[x] = gsl_rng_get(rng) ;
	
	#pragma omp parallel default(shared)
	{
		int d, t, rank = omp_get_thread_num() ;
		double quat[4] ;
		double *view = malloc(det->num_pix * sizeof(double)) ;
		gsl_rng *rng = gsl_rng_alloc(T) ;
		gsl_rng_set(rng, seeds[rank]) ;
		
		#pragma omp for schedule(static) reduction(+:intens_ave)
		for (d = 0 ; d < NUM_AVE ; ++d) {
			if (num_rot == 0) {
				rand_quat(quat, rng) ;
				slice_gen3d(quat, 0., view, intens, size, det) ;
			}
			else {
				slice_gen3d(&quat_list[4*gsl_rng_uniform_int(rng, num_rot)], 0., view, intens, size, det) ;
			}
			
			for (t = 0 ; t < det->num_pix ; ++t){
				if (det->mask[t] > 1)
					continue ;
				intens_ave += view[t] ;
			}
		}
		
		free(view) ;
		gsl_rng_free(rng) ;
	}
	
	free(seeds) ;
	gsl_rng_free(rng) ;
	intens_ave /= NUM_AVE ;
	
    if (scale_method == FLUENCE) {
		rescale = fluence*pow(2.81794e-9, 2) ;
		mean_count = rescale*intens_ave ;
		fprintf(stderr, "Target mean_count = %f for fluence = %.3e photons/um^2\n", mean_count, fluence) ;
	}
	else if (scale_method == COUNTS)
		rescale = mean_count / intens_ave ;
	
	for (x = 0 ; x < size * size * size ; ++x)
		intens[x] *= rescale ;
}

void allocate_data_memory() {
	int d ;
	long num_ones, num_multi ;
	
	ones = calloc(num_data, sizeof(int)) ;
	multi = calloc(num_data, sizeof(int)) ;
	place_ones = malloc(num_data * sizeof(int*)) ;
	place_multi = malloc(num_data * sizeof(int*)) ;
	count_multi = malloc(num_data * sizeof(int*)) ;
	likelihood = calloc(num_data, sizeof(double)) ;
	scale_factors = malloc(num_data * sizeof(double)) ;
	
	num_multi = (mean_count + background*det->num_pix) > det->num_pix ?
	            det->num_pix :
	            (mean_count + background*det->num_pix) ;
	num_ones = 10*num_multi > det->num_pix ? det->num_pix : 10*num_multi ;
	fprintf(stderr, "Assuming maximum of %ld and %ld ones and multi pixels respectively.\n", num_ones, num_multi) ;
	
	for (d = 0 ; d < num_data ; ++d) {
		place_ones[d] = malloc((long) num_ones * sizeof(int)) ;
		place_multi[d] = malloc((long) num_multi * sizeof(int)) ;
		count_multi[d] = malloc((long) num_multi * sizeof(int)) ;
	}
}

double calc_dataset() {
	int x ;
	double actual_mean_count = 0. ;
	const gsl_rng_type *T = gsl_rng_default ;
	gsl_rng *rng = gsl_rng_alloc(T) ;
	unsigned long *seeds = malloc(omp_get_max_threads() * sizeof(unsigned long)) ;
	
	if (testing_mode) {
		gsl_rng_set(rng, 0x5EED) ;
	}
	else {
		struct timeval tval ;
		gettimeofday(&tval, NULL) ;
		gsl_rng_set(rng, tval.tv_sec + tval.tv_usec) ;
	}
	for (x = 0 ; x < omp_get_max_threads() ; ++x)
		seeds[x] = gsl_rng_get(rng) ;
	
	#pragma omp parallel default(shared)
	{
		int photons, d, t, rank = omp_get_thread_num() ;
		double scale = 1., quat[4], val ;
		double *view = malloc(det->num_pix * sizeof(double)) ;
		gsl_rng *rng = gsl_rng_alloc(T) ;
		gsl_rng_set(rng, seeds[rank]) ;
		
		#pragma omp for schedule(static,1) reduction(+:actual_mean_count)
		for (d = 0 ; d < num_data ; ++d) {
			if (num_rot == 0) {
				rand_quat(quat, rng) ;
				slice_gen3d(quat, 0., view, intens, size, det) ;
			}
			else {
				slice_gen3d(&quat_list[4*gsl_rng_uniform_int(rng, num_rot)], 0., view, intens, size, det) ;
			}
			
			if (do_gamma)
				scale = gsl_ran_gamma(rng, 2., 0.5) ;
			
			if (scale > 0.) {
				for (t = 0 ; t < det->num_pix ; ++t) {
					if (det->mask[t] > 1)
						continue ;
					
					val = view[t]*scale + background ;
					photons = gsl_ran_poisson(rng, val) ;
					
					if (photons == 1) {
						place_ones[d][ones[d]++] = t ;
					}
					else if (photons > 1) {
						place_multi[d][multi[d]] = t ;
						count_multi[d][multi[d]++] = photons ;
						actual_mean_count += photons ;
					}
					
					if (likelihood_fname[0] != '\0') {
						if (photons == 0)
							likelihood[d] -= val ;
						else
							likelihood[d] += photons*log(val) - val - gsl_sf_lnfact(photons) ;
					}
					if (scale_fname[0] != '\0')
						scale_factors[d] = scale ;
				}
			}
			
			actual_mean_count += ones[d] ;
			
			if (rank == 0)
				fprintf(stderr, "\rFinished d = %d", d) ;
		}
		
		free(view) ;
		gsl_rng_free(rng) ;
	}
	 
	free(seeds) ;
	gsl_rng_free(rng) ;
	fprintf(stderr, "\rFinished d = %d\n", num_data) ;
	return actual_mean_count / num_data ;
}

void write_dataset() {
	int d, header[256] = {0} ;
	header[0] = num_data ;
	header[1] = det->num_pix ;
	
	FILE *fp = fopen(output_fname, "wb") ;
	fwrite(header, sizeof(int), 256, fp) ;
	fwrite(ones, sizeof(int), num_data, fp) ;
	fwrite(multi, sizeof(int), num_data, fp) ;
	for (d = 0 ; d < num_data ; ++d)
		fwrite(place_ones[d], sizeof(int), ones[d], fp) ;
	for (d = 0 ; d < num_data ; ++d)
		fwrite(place_multi[d], sizeof(int), multi[d], fp) ;
	for (d = 0 ; d < num_data ; ++d)
		fwrite(count_multi[d], sizeof(int), multi[d], fp) ;
	fclose(fp) ;
	
	if (likelihood_fname[0] != '\0') {
		fp = fopen(likelihood_fname, "wb") ;
		fwrite(likelihood, sizeof(double), num_data, fp) ;
		fclose(fp) ;
	}
	if (scale_fname[0] != '\0') {
		fp = fopen(scale_fname, "w") ;
		for (d = 0 ; d < num_data ; ++d)
			fprintf(fp, "%13.10f\n", scale_factors[d]) ;
		fclose(fp) ;
	}
}

char *generate_token(char *line, char *section_name) {
	char *token = strtok(line, " =") ;
	if (token[0] == '#' || token[0] == '\n')
		return NULL ;
	
	if (line[0] == '[') {
		token = strtok(line, "[]") ;
		strcpy(section_name, token) ;
		return NULL ;
	}
	
	return token ;
}

int generate_size_params(char *config_fname) {
	double qmin, qmax, hx, hy ;
	double detd = 0., pixsize = 0., ewald_rad = -1. ;
	int detsize = 0, dets_x = 0, dets_y = 0 ;
	char line[1024], section_name[1024], *token ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "parameters") == 0) {
			if (strcmp(token, "detd") == 0)
				detd = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "detsize") == 0) {
				dets_x = atoi(strtok(NULL, " =\n")) ;
				dets_y = dets_x ;
				token = strtok(NULL, " =\n") ;
				if (token == NULL)
					detsize = dets_x ;
				else {
					dets_y = atoi(token) ;
					detsize = dets_x > dets_y ? dets_x : dets_y ;
				}
			}
			else if (strcmp(token, "pixsize") == 0)
				pixsize = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "ewald_rad") == 0)
				ewald_rad = atof(strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	if (detsize == 0 || pixsize == 0. || detd == 0.) {
		fprintf(stderr, "Need detector parameters: detd, detsize, pixsize\n") ;
		return 1 ;
	}
	
	if (det->detd > 0.)
		detd = det->detd ;
	else
		detd /= pixsize ;
	if (det->ewald_rad > 0.)
		ewald_rad = det->ewald_rad ;
	hx = (dets_x - 1) / 2 ;
	hy = (dets_y - 1) / 2 ;
	qmax = 2. * sin(0.5 * atan(sqrt(hx*hx + hy*hy)/detd)) ;
	qmin = 2. * sin(0.5 * atan(1./detd)) ;
	if (ewald_rad == -1.)
		size = 2 * ceil(qmax / qmin) + 3 ;
	else
		size = 2 * ceil(qmax / qmin * ewald_rad / detd) + 3 ;
	fprintf(stderr, "Calculated size of %d voxels from config parameters\n", size) ;
	
	return 0 ;
}

int generate_intens(char *config_fname) {
	FILE *fp ;
	char intens_fname[1024], out_intens_fname[1024] ;
	char line[1024], section_name[1024], *token ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "make_data") == 0) {
			if (strcmp(token, "in_intensity_file") == 0)
				strcpy(intens_fname, strtok(NULL, " =\n")) ;
		}
		else if (strcmp(section_name, "make_intensities") == 0) {
			if (strcmp(token, "out_intensity_file") == 0)
				strcpy(out_intens_fname, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	if (strcmp(intens_fname, "make_intensities:::out_intensity_file") == 0)
		strcpy(intens_fname, out_intens_fname) ;
	
	fp = fopen(intens_fname, "rb") ;
	if (fp == NULL) {
		fprintf(stderr, "in_intensity_file: %s not found.\n", intens_fname) ;
		return 1 ;
	}
	intens = malloc(size * size * size * sizeof(double)) ;
	fread(intens, sizeof(double), size*size*size, fp) ;
	fclose(fp) ;

	return 0 ;
}

int generate_quat_list(char *config_fname) {
	int t ;
	FILE *fp ;
	char quat_fname[1024] = {'\0'} ;
	char line[1024], section_name[1024], *token ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "make_data") == 0) {
			if (strcmp(token, "in_quat_list") == 0)
				strcpy(quat_fname, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	if (quat_fname[0] != '\0') {
		fprintf(stderr, "Picking discrete orientations from %s\n", quat_fname) ;
		fp = fopen(quat_fname, "r") ;
		if (fp == NULL) {
			fprintf(stderr, "Unable to open %s\n", quat_fname) ;
			return 1 ;
		}
		fscanf(fp, "%d\n", &num_rot) ;
		quat_list = malloc(num_rot * 4 * sizeof(double)) ;
		for (t = 0 ; t < num_rot*4 ; ++t)
			fscanf(fp, "%lf", &quat_list[t]) ;
		fclose(fp) ;
	}
	
	return 0 ;
}

int generate_globals(char *config_fname) {
	char line[1024], section_name[1024], *token ;
	
	size = 0 ;
	num_data = 0 ;
	fluence = -1. ;
	mean_count = -1. ;
	do_gamma = 0 ;
	background = 0. ;
	num_rot = 0 ;
	output_fname[0] = '\0' ;
	likelihood_fname[0] = '\0' ;
	scale_fname[0] = '\0' ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "make_data") == 0) {
			if (strcmp(token, "num_data") == 0)
				num_data = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "mean_count") == 0)
				mean_count = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "fluence") == 0)
				fluence = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "bg_count") == 0)
				background = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "gamma_fluence") == 0)
				do_gamma = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "out_photons_file") == 0)
				strcpy(output_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "out_likelihood_file") == 0)
				strcpy(likelihood_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "out_scale_file") == 0)
				strcpy(scale_fname, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;

	// Check for required parameters
	if (num_data == 0) {
		fprintf(stderr, "Need num_data (number of frames to be generated)\n") ;
		return 1 ;
	}
	if (output_fname[0] == '\0') {
		fprintf(stderr, "Need out_photons_file (name of output emc format file)\n") ;
		return 1 ;
	}
	if (fluence < 0.) {
		if (mean_count < 0.) {
			fprintf(stderr, "Need either:\n") ;
			fprintf(stderr, "\tfluence (incident beam intensity in photons/micron^2/pulse)\n") ;
			fprintf(stderr, "\tmean_count (mean number of photons/frame)\n") ;
			return 1 ;
		}
		else {
			scale_method = COUNTS ;
			fprintf(stderr, "Target mean_count = %f\n", mean_count) ;
		}
	}
	else {
		if (mean_count < 0.)
			scale_method = FLUENCE ;
		else {
			fprintf(stderr, "Please specify only one of fluence of mean_count\n") ;
			return 1 ;
		}
	}
	if (likelihood_fname[0] != '\0')
		fprintf(stderr, "Saving frame-by-frame likelihoods to %s\n", likelihood_fname) ;
	if (do_gamma)
		fprintf(stderr, "Assuming Gamma-distributed variable incident fluence\n") ;
	fprintf(stderr, "Parsed global parameters from config file\n") ;
	
	return 0 ;
}

int setup(char *config_fname) {
	FILE *fp ;
	
	fp = fopen(config_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", config_fname) ;
		return 1 ;
	}
	fclose(fp) ;
	if (generate_globals(config_fname))
		return 1 ;
	if (generate_detectors(config_fname, "make_data", &det, 0) < 0.)
		return 1 ;
	background /= det[0].num_pix ;
	if (generate_size_params(config_fname))
		return 1 ;
	if (generate_intens(config_fname))
		return 1 ;
	if (generate_quat_list(config_fname))
		return 1 ;

	return 0 ;
}

void free_mem() {
	int d ;
	
	free(intens) ;
	free_detector(det) ;
	free(likelihood) ;
	
	free(ones) ;
	free(multi) ;
	for (d = 0 ; d < num_data ; ++d) {
		free(place_ones[d]) ;
		free(place_multi[d]) ;
		free(count_multi[d]) ;
	}
	free(place_ones) ;
	free(place_multi) ;
	free(count_multi) ;
}

