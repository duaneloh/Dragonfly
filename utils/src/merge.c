#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_rng.h>

#include "../../src/utils.h"
#include "../../src/detector.h"
#include "../../src/dataset.h"
#include "../../src/interp.h"
#include "../../src/iterate.h"
#include "../../src/quat.h"

struct detector *det ;
struct dataset *frames ;
struct iterate *iter ;
struct rotation *quat ;

int quat_list_from_config(char *config_fname) {
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
	
	if (parse_quat(quat_fname, 0, quat) < 0)
		return 1 ;
	if (quat->num_rot < frames->tot_num_data) {
		fprintf(stderr, "Number of quaternions and frames do not match\n") ;
		return 1 ;
	}
	if (invert_quat) {
		for (r = 0 ; r < quat->num_rot ; ++r)
		for (t = 1 ; t < 4 ; ++t)
			quat->quat[r*4 + t] *= -1. ;
	}
	
	fprintf(stderr, "First quat->quat = (%.3f, %.3f, %.3f, %.3f)\n", quat->quat[0], quat->quat[1], quat->quat[2], quat->quat[3]) ;
	fprintf(stderr, "Second quat->quat = (%.3f, %.3f, %.3f, %.3f)\n", quat->quat[5], quat->quat[6], quat->quat[7], quat->quat[8]) ;
	
	return 0 ;
}

int globals_from_config(char *config_fname, char *output_fname, char *scale_fname) {
	char line[1024], section_name[1024], *token ;
	
	frames = calloc(1, sizeof(struct dataset)) ;
	iter = calloc(1, sizeof(struct iterate)) ;
	quat = calloc(1, sizeof(struct rotation)) ;
	
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
			else if (strcmp(token, "scale_file") == 0)
				strcpy(scale_fname, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	if (output_fname[0] == '\0') {
		fprintf(stderr, "out_merge_file not specified.\n") ;
		return 1 ;
	}
	
	return 0 ;
}

int rel_quat_from_config(char *config_fname) {
	char line[1024], section_name[1024], *token ;
	char probs_fname[1024] ;
	int num_div = -1 ;
	
	probs_fname[0] = '\0' ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "merge") == 0) {
			if (strcmp(token, "in_probs_file") == 0)
				strcpy(probs_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "num_div") == 0)
				num_div = atoi(strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
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

int setup(char *fname, char *output_fname) {
	double qmax = -1 ;
	FILE *fp ;
	char scale_fname[1024] ;
	
	fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", fname) ;
		return 1 ;
	}
	fclose(fp) ;
	if (globals_from_config(fname, output_fname, scale_fname))
		return 1 ;
	if ((qmax = detector_from_config(fname, "merge", &det, 1)) < 0.)
		return 1 ;
	calculate_size(qmax, iter) ;
	if (data_from_config(fname, "merge", "in", det, frames))
		return 1 ;
	iter->tot_num_data = frames->tot_num_data ;
	iter->scale = malloc(iter->tot_num_data * sizeof(double)) ;
	if (quat_list_from_config(fname) && rel_quat_from_config(fname))
		return 1 ;
	
	parse_scale(scale_fname, iter->scale, iter) ;
	
	return 0 ;
}

int main(int argc, char *argv[]) {
	long i, vol ;
	FILE *fp ;
	char config_fname[1024], output_fname[1024] ;
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
	if (setup(config_fname, output_fname))
		return 1 ;
	iter->center = iter->size / 2 ;
	vol = (long)iter->size*iter->size*iter->size ;
	iter->model1 = calloc(vol, sizeof(double)) ;
	iter->inter_weight = calloc(vol, sizeof(double)) ;

	#pragma omp parallel default(shared)
	{
		int omp_rank = omp_get_thread_num() ;
		long detn, curr_d, d, t, r, dset = 0, old_detn = -1 ;
		double *pview = NULL ;
		double *priv_model = calloc(vol, sizeof(double)) ;
		double *priv_weight = calloc(vol, sizeof(double)) ;
		double *view = malloc(det->num_pix * sizeof(double)) ;
		struct dataset *curr = frames ;
		if (iter->rel_quat != NULL)
			pview = malloc(det->num_pix * sizeof(double)) ;
		
		while (curr != NULL) {
			detn = det[0].mapping[dset] ;
			if (detn != old_detn) {
				free(view) ;
				view = malloc(det[detn].num_pix*sizeof(double)) ;
				old_detn = detn ;
			}
			
			#pragma omp for schedule(static,1)
			for (curr_d = 0 ; curr_d < curr->num_data ; ++curr_d) {
				d = curr->num_data_prev + curr_d ;
				
				memset(view, 0, det[detn].num_pix * sizeof(double)) ;
				for (t = 0 ; t < curr->ones[curr_d] ; ++t)
					view[curr->place_ones[curr->ones_accum[curr_d]+t]] += 1 ;
				for (t = 0 ; t < curr->multi[curr_d] ; ++t)
					view[curr->place_multi[curr->multi_accum[curr_d]+t]] += curr->count_multi[curr->multi_accum[curr_d]+t] ;
				
				for (t = 0 ; t < det[detn].num_pix ; ++t)
					view[t] *= iter->scale[d] ;
				
				if (iter->rel_quat == NULL) {
					slice_merge3d(&quat->quat[5*d], view, priv_model, priv_weight, iter->size, &det[detn]) ;
				}
				else {
					for (r = 0 ; r < iter->num_rel_quat[d] ; ++r) {
						for (t = 0 ; t < det[detn].num_pix ; ++t)
							pview[t] = view[t] * iter->rel_prob[d][r] ;
						slice_merge3d(&quat->quat[5*iter->rel_quat[d][r]], pview, priv_model, priv_weight, iter->size, &det[detn]) ;
					}
				}
				
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
			for (t = 0 ; t < vol ; ++t) {
				iter->model1[t] += priv_model[t] ;
				iter->inter_weight[t] += priv_weight[t] ;
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
	free_detector(det) ;
	free_quat(quat) ;
	free_data(0, frames) ;
	
	return 0 ;
}

