#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <hdf5.h>

#include "../../src/detector.h"
#include "../../src/dataset.h"
#include "../../src/interp.h"

int size, center ;
struct detector *det ;
struct dataset *frames ;
double *quat ;
char output_fname[999], dset_name[999] ;
char (*file_list)[999] ;

int parse_quat(char *fname, int invert_quat) {
	int r, t ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "quaternion file %s not found. Exiting.\n", fname) ;
		return 1 ;
	}
	quat = malloc(frames->tot_num_data * 4 * sizeof(double)) ;
	for (r = 0 ; r < frames->tot_num_data ; ++r) {
		fscanf(fp, "%lf ", &quat[r*4]) ;
		for (t = 1 ; t < 4 ; ++t) {
			fscanf(fp, "%lf ", &quat[r*4 + t]) ;
			if (invert_quat)
				quat[r*4 + t] = -quat[r*4 + t] ;
		}
	}
	fclose(fp) ;
	
	fprintf(stderr, "First quat = (%.3f, %.3f, %.3f, %.3f)\n", quat[0], quat[1], quat[2], quat[3]) ;
	fprintf(stderr, "Second quat = (%.3f, %.3f, %.3f, %.3f)\n", quat[4], quat[5], quat[6], quat[7]) ;
	
	return 0 ;
}

int setup(char *fname) {
	int detsize, dets_x, dets_y, invert_quat ;
	double detd, pixsize, ewald_rad, qmin, qmax ;
	FILE *fp ;
	char line[999], *token ;
	char det_fname[999], out_det_fname[999] ;
	char quat_fname[999], data_fname[999] ;
	char out_data_fname[999], data_flist[999] ;
	char section_name[1024] ;

	detsize = 0 ;
	dets_x = 0 ;
	dets_y = 0 ;
	pixsize = 0. ;
	detd = 0. ;
	ewald_rad = -1. ;
	invert_quat = 0 ;
	det_fname[0] = '\0' ;
	out_det_fname[0] = '\0' ;
	quat_fname[0] = '\0' ;
	output_fname[0] = '\0' ;
	data_fname[0] = '\0' ;
	data_flist[0] = '\0' ;

	fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", fname) ;
		return 1 ;
	}
	while (fgets(line, 999, fp) != NULL) {
		token = strtok(line, " =") ;
		if (token[0] == '#' || token[0] == '\n') {
			continue ;
		}
		else if (token[0] == '[') {
			token = strtok(token, "[]") ;
			strcpy(section_name, token) ;
			continue ;
		}
		
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
		else if (strcmp(section_name, "merge") == 0) {
			if (strcmp(token, "in_photons_file") == 0)
				strcpy(data_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "out_photons_file") == 0)
				strcpy(out_data_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_photons_list") == 0)
				strcpy(data_flist, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_detector_file") == 0)
				strcpy(det_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "out_detector_file") == 0)
				strcpy(out_det_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_quat_file") == 0)
				strcpy(quat_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "invert_quat") == 0)
				invert_quat = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "out_merge_file") == 0)
				strcpy(output_fname, strtok(NULL, " =\n")) ;
			else
				fprintf(stderr, "Unknown parameter in [merge]: %s\n", token) ;
		}
	}

	if (strcmp(det_fname, "make_detector:::out_detector_file") == 0)
		strcpy(det_fname, out_det_fname) ;
	if (strcmp(data_fname, "make_data:::out_photons_file") == 0)
		strcpy(data_fname, out_data_fname) ;
	
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

	det = malloc(sizeof(struct detector)) ;
	if (parse_detector(det_fname, det, 1))
		return 1 ;

	frames = malloc(sizeof(struct dataset)) ;
	frames->next = NULL ;
	if (data_flist[0] != '\0' && data_fname[0] != '\0') {
		fprintf(stderr, "Config file contains both in_photons_file and in_photons_list. Pick one.\n") ;
		return 1 ;
	}
	else if (data_flist[0] == '\0') {
		if (parse_dataset(data_fname, det, frames))
			return 1 ;
	}
	else if (parse_data(data_flist, det, frames))
		return 1 ;
	
	if (parse_quat(quat_fname, invert_quat))
		return 1 ;
	
	return 0 ;
}

int main(int argc, char *argv[]) {
	long i, vol ;
	FILE *fp ;
	struct dataset *curr ;
	char config_fname[999] ;
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
				fprintf(stderr, "format: %s [-c config_fname] [-t num_threads] [-h]\n", argv[0]) ;
				return 1 ;
		}
	}
	
	fprintf(stderr, "Generating merge with parameters from %s\n", config_fname) ;
	if (setup(config_fname))
		return 1 ;
	center = size / 2 ;
	vol = (long)size*size*size ;
	double *model = calloc(vol, sizeof(double)) ;
	double *weight = calloc(vol, sizeof(double)) ;

	#pragma omp parallel default(shared)
	{
		int omp_rank = omp_get_thread_num() ;
		long d, t, i ;
		double *priv_model = calloc(vol, sizeof(double)) ;
		double *priv_weight = calloc(vol, sizeof(double)) ;
		double *view = malloc(det->num_pix * sizeof(double)) ;
		
		if (omp_rank == 0)
			curr = frames ;
		#pragma omp barrier
		
		while (curr != NULL) {
			#pragma omp for schedule(static,1)
			for (d = 0 ; d < curr->num_data ; ++d) {
				memset(view, 0, det->num_pix * sizeof(double)) ;
				for (t = 0 ; t < curr->ones[d] ; ++t)
					view[curr->place_ones[curr->ones_accum[d]+t]] += 1 ;
				for (t = 0 ; t < curr->multi[d] ; ++t)
					view[curr->place_multi[curr->multi_accum[d]+t]] += curr->count_multi[curr->multi_accum[d]+t] ;
				
				slice_merge(&quat[4*d], view, priv_model, priv_weight, size, det) ;
				if (omp_rank == 0)
					fprintf(stderr, "\rMerging %s : %ld/%d", curr->filename, d+1, curr->num_data) ;
			}
			if (omp_rank == 0)
				fprintf(stderr, "\rMerging %s : %d/%d done\n", curr->filename, curr->num_data, curr->num_data) ;
			
			if (omp_rank == 0)
				curr = curr->next ;
			#pragma omp barrier
		}
		
		#pragma omp critical(model)
		{
			for (i = 0 ; i < vol ; ++i) {
				model[i] += priv_model[i] ;
				weight[i] += priv_weight[i] ;
			}
		}
		
		free(priv_model) ;
		free(priv_weight) ;
		free(view) ;
	}

	for (i = 0 ; i < vol ; ++i)
	if (weight[i] > 0.)
		model[i] /= weight[i] ;
	symmetrize_friedel(model, size) ;
	
	fp = fopen(output_fname, "wb") ;
	fwrite(model, sizeof(double), vol, fp) ;
	fclose(fp) ;
	
	fp = fopen("data/weights.bin", "wb") ;
	fwrite(weight, sizeof(double), vol, fp) ;
	fclose(fp) ;
	
	fprintf(stderr, "Saved %d-cubed model to %s\n", size, output_fname) ;
	
	free(model) ;
	free(weight) ;
	free(quat) ;
	free_detector(det) ;
	free_data(0, frames) ;
	
	return 0 ;
}

