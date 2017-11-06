#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_rng.h>

#include "../../src/detector.h"
#include "../../src/dataset.h"
#include "../../src/interp.h"

int size, center ;
struct detector *det ;
struct dataset *frames ;
double *quat ;
char output_fname[1024], dset_name[1024] ;
char (*file_list)[1024] ;

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
	int invert_quat ;
	double qmax = -1 ;
	FILE *fp ;
	char line[1024], *token ;
	char det_fname[1024], out_det_fname[1024] ;
	char quat_fname[1024], data_fname[1024] ;
	char out_data_fname[1024], data_flist[1024] ;
	char det_flist[1024], section_name[1024] ;

	invert_quat = 0 ;
	det_fname[0] = '\0' ;
	det_flist[0] = '\0' ;
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
	while (fgets(line, 1024, fp) != NULL) {
		token = strtok(line, " =") ;
		if (token[0] == '#' || token[0] == '\n') {
			continue ;
		}
		else if (token[0] == '[') {
			token = strtok(token, "[]") ;
			strcpy(section_name, token) ;
			continue ;
		}
		
		if (strcmp(section_name, "merge") == 0) {
			if (strcmp(token, "size") == 0)
				size = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_photons_file") == 0)
				strcpy(data_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_photons_list") == 0)
				strcpy(data_flist, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_detector_file") == 0)
				strcpy(det_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_detector_list") == 0)
				strcpy(det_flist, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_quat_file") == 0)
				strcpy(quat_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "out_merge_file") == 0)
				strcpy(output_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "invert_quat") == 0)
				invert_quat = atoi(strtok(NULL, " =\n")) ;
			else
				fprintf(stderr, "Unknown parameter in [merge]: %s\n", token) ;
		}
		else if (strcmp(section_name, "make_detector") == 0) {
			if (strcmp(token, "out_detector_file") == 0)
				strcpy(out_det_fname, strtok(NULL, " =\n")) ;
		}
		else if (strcmp(section_name, "make_data") == 0) {
			if (strcmp(token, "out_photons_file") == 0)
				strcpy(out_data_fname, strtok(NULL, " =\n")) ;
		}
	}

	// Parse detector(s)
	if (det_flist[0] != '\0' && det_fname[0] != '\0') {
		fprintf(stderr, "Both in_detector_file and in_detector_list specified. Pick one.\n") ;
		return 1 ;
	}
	else if (det_fname[0] != '\0') {
		det = malloc(sizeof(struct detector)) ;
		det[0].num_det = 1 ;
		memset(det[0].mapping, 0, 1024*sizeof(int)) ;
		if ((qmax = parse_detector(det_fname, det, 1)) < 0.)
			return 1 ;
	}
	else if (det_flist[0] != '\0') {
		if ((qmax = parse_detector_list(det_flist, &det, 1)) < 0.)
			return 1 ;
	}
	else {
		fprintf(stderr, "Need either in_detector_file or in_detector_list.\n") ;
		return 1 ;
	}

	// Get volume size
	if (size < 0) {
		size = 2 * ceil(qmax) + 3 ;
		fprintf(stderr, "Calculated volume size = %d\n", size) ;
	}
	else {
		fprintf(stderr, "Provided volume size = %d\n", size) ;
	}
	center = size / 2 ;

	// Parse dataset(s)
	int num_datasets ;
	frames = malloc(sizeof(struct dataset)) ;
	frames->next = NULL ;
	if (data_flist[0] != '\0' && data_fname[0] != '\0') {
		fprintf(stderr, "Config file contains both in_photons_file and in_photons_list. Pick one.\n") ;
		return 1 ;
	}
	else if (data_flist[0] == '\0') {
		if (parse_dataset(data_fname, det, frames))
			return 1 ;
		num_datasets = 1 ;
	}
	else if ((num_datasets = parse_data(data_flist, det, frames)) < 0) {
		return 1 ;
	}
	else {
		fprintf(stderr, "Need either in_photons_file or in_photons_list.\n") ;
		return 1 ;
	}
	if (det[0].num_det != num_datasets) {
		fprintf(stderr, "Number of detector files and emc files don't match (%d vs %d)\n", det[0].num_det, num_datasets) ;
		return 1 ;
	}

	// Parse orientations
	if (parse_quat(quat_fname, invert_quat))
		return 1 ;
	
	return 0 ;
}

int main(int argc, char *argv[]) {
	long i, vol ;
	FILE *fp ;
	struct dataset *curr ;
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
		long detn, d, t, i, dset = 0 ;
		double *priv_model = calloc(vol, sizeof(double)) ;
		double *priv_weight = calloc(vol, sizeof(double)) ;
		double *view = malloc(det->num_pix * sizeof(double)) ;
		
		if (omp_rank == 0)
			curr = frames ;
		#pragma omp barrier
		
		while (curr != NULL) {
			detn = det[0].mapping[dset] ;
			realloc(view, det[detn].num_pix*sizeof(double)) ;
			
			#pragma omp for schedule(static,1)
			for (d = 0 ; d < curr->num_data ; ++d) {
				memset(view, 0, det[detn].num_pix * sizeof(double)) ;
				for (t = 0 ; t < curr->ones[d] ; ++t)
					view[curr->place_ones[curr->ones_accum[d]+t]] += 1 ;
				for (t = 0 ; t < curr->multi[d] ; ++t)
					view[curr->place_multi[curr->multi_accum[d]+t]] += curr->count_multi[curr->multi_accum[d]+t] ;
				
				slice_merge(&quat[4*d], view, priv_model, priv_weight, size, &det[detn]) ;
				if (omp_rank == 0)
					fprintf(stderr, "\rMerging %s : %ld/%d", curr->filename, d+1, curr->num_data) ;
			}
			if (omp_rank == 0)
				fprintf(stderr, "\rMerging %s : %d/%d done\n", curr->filename, curr->num_data, curr->num_data) ;
			
			if (omp_rank == 0)
				curr = curr->next ;
			dset++ ;
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

