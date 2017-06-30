#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <inttypes.h>
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

double rot[2][2] ;
int size, num_data, num_data_p, scale_method, num_rot, do_gamma ;
int **place_ones, **place_multi, *ones, *multi, **count_multi ;
double fluence, rescale, mean_count, detd, back, center ;
double *intens, *likelihood, *quat_list ;
char output_fname[999], likelihood_fname[999] ;
struct detector *det ;

int setup(char *) ;
void rand_quat(double[4], gsl_rng*) ;
int poisson(double, gsl_rng*) ;
double rand_scale() ;
void free_mem() ;

int main(int argc, char *argv[]) {
	int c, d, x ;
	double intens_ave, actual_mean_count = 0. ;
	FILE *fp ;
	struct timeval t1, t2 ;
	
	char config_fname[999] ;
	extern char *optarg ;
	extern int optind ;
	
	omp_set_num_threads(omp_get_max_threads()) ;
	strcpy(config_fname, "config.ini") ;
	
	while ((c = getopt(argc, argv, "c:t:h")) != -1) {
		switch (c) {
			case 't':
				omp_set_num_threads(atoi(optarg)) ;
				break ;
			case 'c':
				strcpy(config_fname, optarg) ;
				break ;
			case 'h':
				fprintf(stderr, "Format: %s [-c config_fname] [-t num_threads] [-h]\n", argv[0]) ;
				return 1 ;
		}
	}
	
	fprintf(stderr, "Generating data with parameters from %s\n", config_fname) ;
	
	if (setup(config_fname))
		return 2 ;
	fprintf(stderr, "Completed setup()\n") ;
	
	gettimeofday(&t1, NULL) ;
	intens_ave = 0. ;
	
	const gsl_rng_type *T ;
	gsl_rng_env_setup() ;
	T = gsl_rng_default ;
	
	#pragma omp parallel default(shared)
	{
		int rank = omp_get_thread_num() ;
		double quat[4] ;
		int d, t ;
		double *view = malloc(det->num_pix * sizeof(double)) ;
		gsl_rng *rng ;
		
		gettimeofday(&t2, NULL) ;
		rng = gsl_rng_alloc(T) ;
		gsl_rng_set(rng, t2.tv_sec + t2.tv_usec + rank) ;
		
		#pragma omp for schedule(static) reduction(+:intens_ave)
		for (d = 0 ; d < NUM_AVE ; ++d) {
			if (num_rot == 0) {
				rand_quat(quat, rng) ;
				slice_gen(quat, 0., view, intens, size, det) ;
			}
			else {
				slice_gen(&quat_list[4*gsl_rng_uniform_int(rng, num_rot)], 0., view, intens, size, det) ;
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
	fprintf(stderr, "Calculated average\n") ;
	
	intens_ave /= NUM_AVE ;
    if (scale_method == FLUENCE) {
		rescale = fluence*pow(2.81794e-9, 2) ;
		mean_count = rescale*intens_ave ;
		fprintf(stderr, "Target mean_count = %f for fluence = %.3e photons/um^2\n", mean_count, fluence) ;
	}
	else if (scale_method == COUNTS)
		rescale = mean_count / intens_ave ;
	
	long num_multi = (mean_count + back*det->num_pix) > det->num_pix ?
	                 det->num_pix :
	                 (mean_count + back*det->num_pix) ;
	long num_ones = 10*num_multi > det->num_pix ? det->num_pix : 10*num_multi ;
	fprintf(stderr, "Assuming maximum of %ld and %ld ones and multi pixels respectively.\n", num_ones, num_multi) ;
	
	for (d = 0 ; d < num_data ; ++d) {
		place_ones[d] = malloc((long) num_ones * sizeof(int)) ;
		place_multi[d] = malloc((long) num_multi * sizeof(int)) ;
		count_multi[d] = malloc((long) num_multi * sizeof(int)) ;
	}
	
	for (x = 0 ; x < size * size * size ; ++x)
		intens[x] *= rescale ;
	
	#pragma omp parallel default(shared)
	{
		int rank = omp_get_thread_num() ;
		int photons, d, t ;
		double scale = 1., quat[4], val ;
		double *view = malloc(det->num_pix * sizeof(double)) ;
		gsl_rng *rng ;
		
		gettimeofday(&t2, NULL) ;
		rng = gsl_rng_alloc(T) ;
		gsl_rng_set(rng, t2.tv_sec + t2.tv_usec + rank) ;
		
		#pragma omp for schedule(static,1) reduction(+:actual_mean_count)
		for (d = 0 ; d < num_data ; ++d) {
			if (num_rot == 0) {
				rand_quat(quat, rng) ;
				slice_gen(quat, 0., view, intens, size, det) ;
			}
			else {
				slice_gen(&quat_list[4*gsl_rng_uniform_int(rng, num_rot)], 0., view, intens, size, det) ;
			}
			
			if (do_gamma)
				scale = gsl_ran_gamma(rng, 2., 0.5) ;
			
			if (scale > 0.) {
				for (t = 0 ; t < det->num_pix ; ++t) {
					if (det->mask[t] > 1)
						continue ;
					
					val = view[t]*scale + back ;
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
				}
			}
			
			actual_mean_count += ones[d] ;

			if (rank == 0)
				fprintf(stderr, "\rFinished d = %d", d) ;
		}
		
		free(view) ;
		gsl_rng_free(rng) ;
	}
	
	fprintf(stderr, "\rFinished d = %d\n", num_data) ;
	actual_mean_count /= num_data ;
	
	fp = fopen(output_fname, "wb") ;
	fwrite(&num_data, sizeof(int), 1, fp) ;
	fwrite(&det->num_pix, sizeof(int), 1, fp) ;
	char buffer[1016] = {0} ;
	fwrite(buffer, sizeof(char), 1016, fp) ;
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
	
	gettimeofday(&t2, NULL) ;
	fprintf(stderr, "Generated %d frames with %f photons/frame\n", num_data, actual_mean_count) ;
	fprintf(stderr, "Time taken = %f s\n", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;
	
	//free_mem() ;
	
	return 0 ;
}

int setup(char *config_fname) {
	int t ;
	FILE *fp ;
	char line[999], *token ;
	char det_fname[999], model_fname[999] ;
	char out_det_fname[999], out_model_fname[999] ;
	char quat_fname[999] ;
	double detd, pixsize, qmax, qmin, ewald_rad ;
	int detsize, dets_x, dets_y ;
	
	size = 0 ;
	center = 0 ;
	num_data = 0 ;
	fluence = -1. ;
	mean_count = -1. ;
	do_gamma = 0 ;
	back = 0. ;
	output_fname[0] = '\0' ;
	detsize = 0 ;
	dets_x = 0 ;
	dets_y = 0 ;
	detd = 0. ;
	pixsize = 0. ;
	ewald_rad = -1. ;
	quat_fname[0] = '\0' ;
	num_rot = 0 ;
	
	fp = fopen(config_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", config_fname) ;
		return 1 ;
	}
	while (fgets(line, 999, fp) != NULL) {
		token = strtok(line, " =") ;
		if (token[0] == '#' || token[0] == '\n' || token[0] == '[')
			continue ;
		
		if (strcmp(token, "num_data") == 0)
			num_data = atoi(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "detd") == 0)
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
		else if (strcmp(token, "mean_count") == 0)
			mean_count = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "fluence") == 0)
			fluence = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "bg_count") == 0)
			back = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "out_photons_file") == 0)
			strcpy(output_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "out_likelihood_file") == 0)
			strcpy(likelihood_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "in_intensity_file") == 0)
			strcpy(model_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "in_detector_file") == 0)
			strcpy(det_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "out_intensity_file") == 0)
			strcpy(out_model_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "out_detector_file") == 0)
			strcpy(out_det_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "in_quat_list") == 0)
			strcpy(quat_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "gamma_fluence") == 0)
			do_gamma = atoi(strtok(NULL, " =\n")) ;
	}
	fclose(fp) ;
	
	if (strcmp(model_fname, "make_intensities:::out_intensity_file") == 0)
		strcpy(model_fname, out_model_fname) ;
	if (strcmp(det_fname, "make_detector:::out_detector_file") == 0)
		strcpy(det_fname, out_det_fname) ;
	
	if (detsize == 0 || pixsize == 0. || detd == 0.) {
		fprintf(stderr, "Need detector parameters: detd, detsize, pixsize\n") ;
		return 1 ;
	}
	
	double hx = (dets_x - 1) / 2 * pixsize ;
	double hy = (dets_y - 1) / 2 * pixsize ;
	qmax = 2. * sin(0.5 * atan(sqrt(hx*hx + hy*hy)/detd)) ;
	qmin = 2. * sin(0.5 * atan(pixsize/detd)) ;
	if (ewald_rad == -1.)
		size = 2 * ceil(qmax / qmin) + 3 ;
	else
		size = 2 * ceil(qmax / qmin * ewald_rad * pixsize / detd) + 3 ;
	center = size / 2 ;
	
	if (num_data == 0) {
		fprintf(stderr, "Need num_data (number of frames to be generated)\n") ;
		return 1 ;
	}
	if (output_fname[0] == '\0') {
		fprintf(stderr, "Need out_photons (name of output emc format file)\n") ;
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
	
	char *config_folder = dirname(config_fname) ;
	strcpy(line, det_fname) ;
	sprintf(det_fname, "%s/%s", config_folder, line) ;
	strcpy(line, output_fname) ;
	sprintf(output_fname, "%s/%s", config_folder, line) ;
	strcpy(line, model_fname) ;
	sprintf(model_fname, "%s/%s", config_folder, line) ;
	
	fp = fopen(model_fname, "rb") ;
	if (fp == NULL) {
		fprintf(stderr, "model_fname: %s not found. Exiting...\n", model_fname) ;
		return 1 ;
	}
	intens = malloc(size * size * size * sizeof(double)) ;
	fread(intens, sizeof(double), size*size*size, fp) ;
	fclose(fp) ;
	
	det = malloc(sizeof(struct detector)) ;
	parse_detector(det_fname, det, 0) ;
	
	if (quat_fname[0] != '\0') {
		fprintf(stderr, "Picking discrete orientations from %s\n", quat_fname) ;
		fp = fopen(quat_fname, "r") ;
		fscanf(fp, "%d\n", &num_rot) ;
		quat_list = malloc(num_rot * 4 * sizeof(double)) ;
		for (t = 0 ; t < num_rot*4 ; ++t)
			fscanf(fp, "%lf", &quat_list[t]) ;
		fclose(fp) ;
	}
	
	back /= det->num_pix ;
	
	ones = calloc(num_data, sizeof(int)) ;
	multi = calloc(num_data, sizeof(int)) ;
	place_ones = malloc(num_data * sizeof(int*)) ;
	place_multi = malloc(num_data * sizeof(int*)) ;
	count_multi = malloc(num_data * sizeof(int*)) ;
	likelihood = calloc(num_data, sizeof(double)) ;
	
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
	fprintf(stderr, "Freeing data memory\n") ;
	free(place_ones) ;
	free(place_multi) ;
	free(count_multi) ;
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

