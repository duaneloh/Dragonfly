#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <mpi.h>
#include "emc.h"

int setup(char *s_config_fname, int continue_flag) {
	FILE *fp ;
	double qmax = -1. ;
	struct timeval t1, t2 ;
	int det_flag ;
	
	gettimeofday(&t1, NULL) ;

	param = malloc(sizeof(struct params)) ;
	iter = malloc(sizeof(struct iterate)) ;
	quat = malloc(sizeof(struct rotation)) ;
	frames = malloc(sizeof(struct dataset)) ;
	merge_frames = NULL ;
	char config_fname[PATH_MAX] ;
	realpath(s_config_fname, config_fname) ;
	MPI_Comm_size(MPI_COMM_WORLD, &param->num_proc) ;
	MPI_Comm_rank(MPI_COMM_WORLD, &param->rank) ;
	
	fp = fopen(config_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", config_fname) ;
		return 1 ;
	}
	fclose(fp) ;
	generate_params(config_fname, param) ;
	generate_output_dirs(param) ;
	if (param->recon_type == RECON3D) {
		det_flag = 1 ;
		slice_gen = &slice_gen3d ;
		slice_merge = &slice_merge3d ;
	}
	else if (param->recon_type == RECON2D) {
		det_flag = -param->modes ;
		slice_gen = &slice_gen2d ;
		slice_merge = &slice_merge2d ;
	}
	else {
		fprintf(stderr, "recon_type not recognized\n") ;
		return 1 ;
	}
	if ((qmax = generate_detectors(config_fname, "emc", &det, det_flag)) < 0.)
		return 1 ;
	if (generate_quaternion(config_fname, "emc", quat))
		return 1 ;
	divide_quat(param->rank, param->num_proc, param->modes, quat) ;
	if (generate_data(config_fname, "emc", "in", det, frames))
		return 1 ;
	if (generate_data(config_fname, "emc", "merge", det, merge_frames))
		return 1 ;
	generate_blacklist(config_fname, frames) ;
	if (generate_iterate(config_fname, "emc", continue_flag, qmax, param, det, frames, iter))
		return 1 ;

	gettimeofday(&t2, NULL) ;
	fprintf(stderr, "Completed setup: %f s\n", (double)(t2.tv_sec - t1.tv_sec) + 1.e-6*(t2.tv_usec - t1.tv_usec)) ;
	
	return 0 ;
}

void free_mem() {
	free_iterate(iter) ;
	iter = NULL ;
	if (merge_frames != NULL) {
		free_data(param->need_scaling, merge_frames) ;
		merge_frames = NULL ;
	}
	free_data(param->need_scaling, frames) ;
	frames = NULL ;
	free_quat(quat) ;
	quat = NULL ;
	free_detector(det) ;
	det = NULL ;
	free(param) ;
	param = NULL ;
}

