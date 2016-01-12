#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <inttypes.h>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <math.h>
#include <sys/time.h>
#include <float.h>
#include <stdint.h>
#include <libgen.h>

#define NUM_AVE 5000
#define FLUENCE 0
#define COUNTS 1

double rot[2][2] ;
int size, num_data, num_data_p, num_pix, scale_method ;
int **place_ones, **place_multi, *ones, *multi, **count_multi ;
double fluence, rescale, mean_count, spread, detd, back, center ;
double *intens, *det, *view ;
char output_fname[999] ;
uint8_t *mask ;

int setup(char *) ;
void rand_quat(double[4], gsl_rng*) ;
int poisson(double, gsl_rng*) ;
double rand_scale() ;
void free_mem() ;
void slice_gen(double*, double*, double*, double*) ;

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
		double *view = malloc(num_pix * sizeof(double)) ;
		gsl_rng *rng ;
		
		gettimeofday(&t2, NULL) ;
		rng = gsl_rng_alloc(T) ;
		gsl_rng_set(rng, t2.tv_sec + t2.tv_usec + rank) ;
		
		#pragma omp for schedule(static) reduction(+:intens_ave)
		for (d = 0 ; d < NUM_AVE ; ++d) {
			rand_quat(quat, rng) ;
			slice_gen(quat, view, intens, det) ;
            
			for (t = 0 ; t < num_pix ; ++t){
				if (mask[t] > 1)
					continue ;
				intens_ave += view[t] ;
            }
		}

		free(view) ;
		gsl_rng_free(rng) ;
	}
	
	intens_ave /= NUM_AVE ;
    if (scale_method == FLUENCE) {
		rescale = fluence*pow(2.81794e-9, 2) ;
		mean_count = rescale*intens_ave ;
		fprintf(stderr, "Target mean_count = %f\n", mean_count) ;
	}
	else if (scale_method == COUNTS)
		rescale = mean_count / intens_ave ;
	
	spread /= mean_count ;
	for (d = 0 ; d < num_data ; ++d) {
		place_ones[d] = malloc((long) 5 * mean_count * (1+spread) * sizeof(int)) ;
		place_multi[d] = malloc((long) mean_count * (1+spread) * sizeof(int)) ;
		count_multi[d] = malloc((long) mean_count * (1+spread) * sizeof(int)) ;
	}
	
	for (x = 0 ; x < size * size * size ; ++x)
		intens[x] *= rescale ;
	
	#pragma omp parallel default(shared)
	{
		int rank = omp_get_thread_num() ;
		int photons, d, t ;
		double scale = 1., quat[4] ;
		double *view = malloc(num_pix * sizeof(double)) ;
		gsl_rng *rng ;
		
		gettimeofday(&t2, NULL) ;
		rng = gsl_rng_alloc(T) ;
		gsl_rng_set(rng, t2.tv_sec + t2.tv_usec + rank) ;
		
		#pragma omp for schedule(static,1) reduction(+:actual_mean_count)
		for (d = 0 ; d < num_data ; ++d) {
			rand_quat(quat, rng) ;
			slice_gen(quat, view, intens, det) ;
			
			if (spread > 0.)
				scale = gsl_ran_gaussian(rng, spread) ;
			
			if (scale > 0.) {
				for (t = 0 ; t < num_pix ; ++t) {
					if (mask[t] > 1)
						continue ;
					
					photons = gsl_ran_poisson(rng, view[t]*scale + back) ;
					
					if (photons == 1)
						place_ones[d][ones[d]++] = t ;
					else if (photons > 1) {
						place_multi[d][multi[d]] = t ;
						count_multi[d][multi[d]++] = photons ;
						actual_mean_count += photons ;
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
	fwrite(&num_pix, sizeof(int), 1, fp) ;
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
	
	gettimeofday(&t2, NULL) ;
	fprintf(stderr, "Generated %d frames with %f photons/frame\n", num_data, actual_mean_count) ;
	fprintf(stderr, "Time taken = %f s\n", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;

	free_mem() ;
	
	return 0 ;
}

int setup(char *config_fname) {
	int t, d ;
	FILE *fp ;
	char line[999], *token ;
	char det_fname[999], model_fname[999] ;
	char out_det_fname[999], out_model_fname[999] ;
	double detd, pixsize, qmax, qmin ;
	int detsize ;
	
	size = 0 ;
	center = 0 ;
	num_data = 0 ;
	fluence = -1. ;
	mean_count = -1. ;
	spread = 0. ;
	back = 0. ;
	output_fname[0] = ' ' ;
	detsize = 0 ;
	detd = 0. ;
	pixsize = 0. ;
	
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
		else if (strcmp(token, "detsize") == 0)
			detsize = atoi(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "pixsize") == 0)
			pixsize = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "mean_count") == 0)
			mean_count = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "fluence") == 0)
			fluence = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "mean_count_spread") == 0)
			spread = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "bg_count") == 0)
			back = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "out_photons_file") == 0)
			strcpy(output_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "in_intensity_file") == 0)
			strcpy(model_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "in_detector_file") == 0)
			strcpy(det_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "out_intensity_file") == 0)
			strcpy(out_model_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "out_detector_file") == 0)
			strcpy(out_det_fname, strtok(NULL, " =\n")) ;
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
	
	qmax = 2. * sin(0.5 * atan(sqrt(2.)*((detsize-1)/2)*pixsize/detd)) ;
	qmin = 2. * sin(0.5 * atan(pixsize/detd)) ;
	size = ceil(2. * qmax / qmin) + 1 ;
	center = size / 2 ;
	
	if (num_data == 0) {
		fprintf(stderr, "Need num_data (number of frames to be generated)\n") ;
		return 1 ;
	}
	if (output_fname[0] == ' ') {
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
	
	fp = fopen(det_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "det_fname: %s not found. Exiting...\n", det_fname) ;
		return 1 ;
	}
	fscanf(fp, "%d", &num_pix) ;
	det = malloc(num_pix * 4 * sizeof(double)) ;
	mask = malloc(num_pix * sizeof(uint8_t)) ;
	for (t = 0 ; t < num_pix ; ++t) {
		for (d = 0 ; d < 4 ; ++d)
			fscanf(fp, "%lf", &det[t*4 + d]) ;
		fscanf(fp, "%" SCNu8, &mask[t]) ;
	}
	fclose(fp) ;
	
	back /= num_pix ;
	view = malloc(num_pix * sizeof(double)) ;
	
	ones = calloc(num_data, sizeof(int)) ;
	multi = calloc(num_data, sizeof(int)) ;
	place_ones = malloc(num_data * sizeof(int*)) ;
	place_multi = malloc(num_data * sizeof(int*)) ;
	count_multi = malloc(num_data * sizeof(int*)) ;
	
	return 0 ;
}

void free_mem() {
	int d ;
	
	free(intens) ;
	free(det) ;
	free(view) ;
	free(mask) ;
	
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

void make_rot_quat(double *quaternion, double rot[3][3]) {
	double q0, q1, q2, q3, q01, q02, q03, q11, q12, q13, q22, q23, q33 ;
	
	q0 = quaternion[0] ;
	q1 = quaternion[1] ;
	q2 = quaternion[2] ;
	q3 = quaternion[3] ;
	
	q01 = q0*q1 ;
	q02 = q0*q2 ;
	q03 = q0*q3 ;
	q11 = q1*q1 ;
	q12 = q1*q2 ;
	q13 = q1*q3 ;
	q22 = q2*q2 ;
	q23 = q2*q3 ;
	q33 = q3*q3 ;
	
	rot[0][0] = (1. - 2.*(q22 + q33)) ;
	rot[0][1] = 2.*(q12 + q03) ;
	rot[0][2] = 2.*(q13 - q02) ;
	rot[1][0] = 2.*(q12 - q03) ;
	rot[1][1] = (1. - 2.*(q11 + q33)) ;
	rot[1][2] = 2.*(q01 + q23) ;
	rot[2][0] = 2.*(q02 + q13) ;
	rot[2][1] = 2.*(q23 - q01) ;
	rot[2][2] = (1. - 2.*(q11 + q22)) ;
}

void slice_gen(double *quaternion, double slice[], double model3d[], double detector[]) {
	int t, i, j, x, y, z ;
	double tx, ty, tz, fx, fy, fz, cx, cy, cz ;
	double rot_pix[3], rot[3][3] = {{0}} ;
	
	make_rot_quat(quaternion, rot) ;
	
	for (t = 0 ; t < num_pix ; ++t) {
		for (i = 0 ; i < 3 ; ++i) {
			rot_pix[i] = 0. ;
			for (j = 0 ; j < 3 ; ++j) 
				rot_pix[i] += rot[i][j] * detector[t*4 + j] ;
			rot_pix[i] += center ;
		}
		
		tx = rot_pix[0] ;
		ty = rot_pix[1] ;
		tz = rot_pix[2] ;
		
		if (tx < 0 || tx > size-2 || ty < 0 || ty > size-2 || tz < 0 || tz > size-2) {
			slice[t] = 1.e-10 ;
			continue ;
		}
		
		x = (int) tx ;
		y = (int) ty ;
		z = (int) tz ;
		fx = tx - x ;
		fy = ty - y ;
		fz = tz - z ;
		cx = 1. - fx ;
		cy = 1. - fy ;
		cz = 1. - fz ;
		
		slice[t] =	cx*cy*cz*model3d[x*size*size + y*size + z] +
				cx*cy*fz*model3d[x*size*size + y*size + ((z+1)%size)] +
				cx*fy*cz*model3d[x*size*size + ((y+1)%size)*size + z] +
				cx*fy*fz*model3d[x*size*size + ((y+1)%size)*size + ((z+1)%size)] +
				fx*cy*cz*model3d[((x+1)%size)*size*size + y*size + z] +
				fx*cy*fz*model3d[((x+1)%size)*size*size + y*size + ((z+1)%size)] + 
				fx*fy*cz*model3d[((x+1)%size)*size*size + ((y+1)%size)*size + z] + 
				fx*fy*fz*model3d[((x+1)%size)*size*size + ((y+1)%size)*size + ((z+1)%size)] ;
		
		// Correct for solid angle and polarization
		slice[t] *= detector[t*4 + 3] ;
	}
}

