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
#include "../../src/iterate.h"

struct detector *det ;
struct dataset *frames ;
struct iterate *iter ;
double *quat ;
char output_fname[1024] ;

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

int generate_quat_list(char *config_fname) {
	int r, t, invert_quat = 0 ;
	char quat_fname[1024] = {'\0'} ;
	char line[1024], section_name[1024], *token ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "merge") == 0) {
			if (strcmp(token, "in_quat_file") == 0)
				strcpy(quat_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "invert_quat") == 0)
				invert_quat = atoi(strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	FILE *fp = fopen(quat_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "in_quat_file %s not found. Exiting.\n", quat_fname) ;
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

int generate_globals(char *config_fname) {
	char line[1024], section_name[1024], *token ;
	
	frames = malloc(sizeof(struct dataset)) ;
	iter = malloc(sizeof(struct iterate)) ;
	
	iter->size = -1 ;
	iter->model2 = NULL ;
	output_fname[0] = '\0' ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "merge") == 0) {
			if (strcmp(token, "size") == 0)
				iter->size = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "out_merge_file") == 0)
				strcpy(output_fname, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	if (output_fname[0] == '\0') {
		fprintf(stderr, "out_merge_file not specified.\n") ;
		return 1 ;
	}
	
	return 0 ;
}

int setup(char *fname) {
	double qmax = -1 ;
	FILE *fp ;
	
	fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", fname) ;
		return 1 ;
	}
	fclose(fp) ;
	if (generate_globals(fname))
		return 1 ;
	if ((qmax = generate_detectors(fname, "merge", &det, 1)) < 0.)
		return 1 ;
	calculate_size(qmax, iter) ;
	if (generate_data(fname, "merge", "in", det, frames))
		return 1 ;
	if (generate_quat_list(fname))
		return 1 ;
	
	return 0 ;
}

int main(int argc, char *argv[]) {
	long i, vol ;
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
				fprintf(stderr, "format: %s [-c config_fname] [-t num_threads] [-h]\n", argv[0]) ;
				return 1 ;
		}
	}
	
	fprintf(stderr, "Generating merge with parameters from %s\n", config_fname) ;
	if (setup(config_fname))
		return 1 ;
	iter->center = iter->size / 2 ;
	vol = (long)iter->size*iter->size*iter->size ;
	iter->model1 = calloc(vol, sizeof(double)) ;
	iter->inter_weight = calloc(vol, sizeof(double)) ;

	#pragma omp parallel default(shared)
	{
		int omp_rank = omp_get_thread_num() ;
		long detn, d, t, i, dset = 0, old_detn = -1 ;
		double *priv_model = calloc(vol, sizeof(double)) ;
		double *priv_weight = calloc(vol, sizeof(double)) ;
		double *view = malloc(det->num_pix * sizeof(double)) ;
		struct dataset *curr = frames ;
		
		while (curr != NULL) {
			detn = det[0].mapping[dset] ;
			if (detn != old_detn) {
				free(view) ;
				view = malloc(det[detn].num_pix*sizeof(double)) ;
				old_detn = detn ;
			}
			
			#pragma omp for schedule(static,1)
			for (d = 0 ; d < curr->num_data ; ++d) {
				memset(view, 0, det[detn].num_pix * sizeof(double)) ;
				for (t = 0 ; t < curr->ones[d] ; ++t)
					view[curr->place_ones[curr->ones_accum[d]+t]] += 1 ;
				for (t = 0 ; t < curr->multi[d] ; ++t)
					view[curr->place_multi[curr->multi_accum[d]+t]] += curr->count_multi[curr->multi_accum[d]+t] ;
				
				slice_merge3d(&quat[4*d], view, priv_model, priv_weight, iter->size, &det[detn]) ;
				if (omp_rank == 0)
					fprintf(stderr, "\rMerging %s : %ld/%d", curr->filename, d+1, curr->num_data) ;
			}
			if (omp_rank == 0)
				fprintf(stderr, "\rMerging %s : %d/%d done\n", curr->filename, curr->num_data, curr->num_data) ;
			
			curr = curr->next ;
			dset++ ;
			#pragma omp barrier
		}
		
		#pragma omp critical(model)
		{
			for (i = 0 ; i < vol ; ++i) {
				iter->model1[i] += priv_model[i] ;
				iter->inter_weight[i] += priv_weight[i] ;
			}
		}
		
		free(priv_model) ;
		free(priv_weight) ;
		free(view) ;
	}

	for (i = 0 ; i < vol ; ++i)
	if (iter->inter_weight[i] > 0.)
		iter->model1[i] /= iter->inter_weight[i] ;
	symmetrize_friedel(iter->model1, iter->size) ;
	
	fp = fopen(output_fname, "wb") ;
	fwrite(iter->model1, sizeof(double), vol, fp) ;
	fclose(fp) ;
	
	fp = fopen("data/weights.bin", "wb") ;
	fwrite(iter->inter_weight, sizeof(double), vol, fp) ;
	fclose(fp) ;
	
	fprintf(stderr, "Saved %ld-cubed model to %s\n", iter->size, output_fname) ;
	
	free_iterate(iter) ;
	free(quat) ;
	free_detector(det) ;
	free_data(0, frames) ;
	
	return 0 ;
}

