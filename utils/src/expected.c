#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_bessel.h>

#include "../../src/utils.h"
#include "../../src/params.h"
#include "../../src/detector.h"
#include "../../src/interp.h"
#include "../../src/iterate.h"
#include "../../src/quat.h"

struct detector *det ;
struct dataset *frames ;
struct iterate *iter ;
struct rotation *quat ;

int generate_rel_quat(char *probs_fname, int num_div) {
	if (probs_fname[0] == '\0') {
		fprintf(stderr, "in_probs_file not specified.\n") ;
		return 1 ;
	}
	if (num_div == -1) {
		fprintf(stderr, "Need num_div with in_probs_file\n") ;
		return 1 ;
	}
	
	if (quat_gen(num_div, quat) < 0) {
		fprintf(stderr, "Problem generating quat[%d]\n", num_div) ;
		return 1 ;
	}
	if (parse_rel_quat(probs_fname, quat->num_rot, 1, iter)) {
		fprintf(stderr, "Problem parsing rel_quat and rel_prob\n") ;
		return 1 ;
	}
	
	return 0 ;
}

int calculate_num_data(char *fname) {
	FILE *fp ;
	char line[8], hdfheader[8] = {137, 'H', 'D', 'F', '\r', '\n', 26, '\n'} ;
	int num_data ;
	
	fp = fopen(fname, "rb") ;
	if (fp == NULL) {
		fprintf(stderr, "Cannot open probs file %s\n", fname) ;
		return -1 ;
	}
	fread(line, sizeof(char), 8, fp) ;
	if (strncmp(line, hdfheader, 8) == 0) {
#ifdef WITH_HDF5
		fclose(fp) ;
		
		hid_t file, dset, dspace ;
		file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT) ;
		dset = H5Dopen(file, "/orientations", H5P_DEFAULT) ;
		dspace = H5Dget_space(dset) ;
		num_data = H5Sget_simple_extent_npoints(dspace) ;
		H5Sclose(dspace) ;
		H5Dclose(dset) ;
		H5Fclose(file) ;
#else // WITH_HDF5
		fprintf(stderr, "H5 output support not compiled. Cannot get tot_num_data\n") ;
		return -1 ;
#endif // WITH_HDF5
	}
	else {
		fseek(fp, 0, SEEK_SET) ;
		fread(&num_data, sizeof(int), 1, fp) ;
		fclose(fp) ;
	}
	
	return num_data ;
}

int setup(char *fname) {
	char line[1024], section_name[1024], *token ;
	char log_fname[1024], output_folder[1024], probs_fname[2048] ;
	double qmax = -1, num_div = -1 ;
	FILE *fp ;
	
	frames = calloc(1, sizeof(struct dataset)) ;
	iter = calloc(1, sizeof(struct iterate)) ;
	quat = calloc(1, sizeof(struct rotation)) ;
	iter->size = -1 ;
	log_fname[0] = '\0' ;
	output_folder[0] = '\0' ;
	probs_fname[0] = '\0' ;
	iter->modes = 1 ; //TODO FIX THIS!!
	
	fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", fname) ;
		return 1 ;
	}
	while (fgets(line, 1024, fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "emc") == 0) {
			if (strcmp(token, "log_file") == 0)
				strcpy(log_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "output_folder") == 0)
				strcpy(output_folder, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_probs_file") == 0)
				strcpy(probs_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "num_div") == 0)
				num_div = atoi(strtok(NULL, " =\n")) ;
		}
	}
	fclose(fp) ;
	
	if (probs_fname[0] == '\0') {
		fp = fopen(log_fname, "r") ;
		if (fp == NULL) {
			fprintf(stderr, "Could not open log file %s\n", log_fname) ;
			return 1 ;
		}
		while(fgets(line, 1024, fp) != NULL) {
			token = strtok(line, " \t\n") ;
			if (token != NULL)
				sprintf(probs_fname, "%s/output_%.3d.h5", output_folder, atoi(token)) ;
		}
		fclose(fp) ;
	}
	fprintf(stderr, "Calculating expected likelihoods using %s\n", probs_fname) ;
	
	if ((qmax = detector_from_config(fname, "emc", &det, 1)) < 0.)
		return 1 ;
	calculate_size(qmax, iter) ;
	iter->tot_num_data = calculate_num_data(probs_fname) ;
	if (generate_rel_quat(probs_fname, num_div))
		return 1 ;
	parse_input(probs_fname, 1., 0, RECON3D, iter) ;
	iter->scale = malloc(iter->tot_num_data * sizeof(double)) ;
	if (!parse_scale(probs_fname, iter->scale, iter))
		return 1 ;
	
	return 0 ;
}

int main(int argc, char *argv[]) {
	FILE *fp ;
	char config_fname[1024] ;
	int c ;
	extern char *optarg ;
	extern int optind ;
	
	omp_set_num_threads(omp_get_max_threads()) ;
	strcpy(config_fname, "config.ini") ;
	
	while ((c = getopt(argc, argv, "c:h")) != -1) {
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
	
	fprintf(stderr, "Config file: %s\n", config_fname) ;
	if (setup(config_fname))
		return 1 ;
	iter->center = iter->size / 2 ;
	double *expect = calloc(iter->tot_num_data, sizeof(double)) ;

	// TODO Update for multiple detectors
	// TODO Update for other recon types
	#pragma omp parallel default(shared)
	{
		int r, d, t, ind ;
		double val ;
		double *view = malloc(det->num_pix * sizeof(double)) ;
		double *priv_expect = calloc(iter->tot_num_data, sizeof(double)) ;
		
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < quat->num_rot ; ++r) {
			slice_gen3d(&(quat->quat[r*5]), 0., view, iter->model1, iter->size, det) ;
			for (d = 0 ; d < iter->tot_num_data ; ++d) {
				// Skip if frame is blacklisted
				if (iter->scale[d] < 0.)
					continue ;
				
				// check if current frame has significant probability
				ind = -1 ;
				for (t = 0 ; t < iter->num_rel_quat[d] ; ++t)
				if (r == iter->rel_quat[d][t]) {
					ind = t ;
					break ;
				}
				if (ind == -1)
					continue ;
				
				for (t = 0 ; t < det->num_pix ; ++t)
				if (det->mask[t] < 1) {
					val = 2. * (view[t] * iter->scale[d] + det->background[t]) ;
					priv_expect[d] += iter->rel_prob[d][ind] * log(gsl_sf_bessel_I0_scaled(val)) ;
					//if (val > priv_expect[d])
					//	priv_expect[d] = val ;
				}
			}
			
			if (r % (quat->num_rot / 100) == 0)
				fprintf(stderr, "\t\tFinished r = %d/%d\n", r, quat->num_rot) ;
		}
		
		#pragma omp critical(expect)
		{
			for (d = 0 ; d < iter->tot_num_data ; ++d)
				expect[d] += priv_expect[d] ;
				//if (priv_expect[d] > expect[d])
				//	expect[d] = priv_expect[d] ;
		}
		
		free(view) ;
		free(priv_expect) ;
	}
	
	fp = fopen("data/expected.bin", "wb") ;
	fwrite(expect, sizeof(double), iter->tot_num_data, fp) ;
	fclose(fp) ;
	fprintf(stderr, "Written frame-by-frame expected likelihoods\n") ;
	
	free_iterate(iter) ;
	free_detector(det) ;
	free_quat(quat) ;
	
	return 0 ;
}

