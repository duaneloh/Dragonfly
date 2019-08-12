#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <mpi.h>
#include "emc.h"

static void backup_log_file(struct params *param) {
	if (access(param->log_fname, F_OK) != -1) {
		char command[4096], copy_fname[1024], backup_fname[2048] ;
		int i = 1 ;
		
		strcpy(copy_fname, param->log_fname) ;
		sprintf(backup_fname, "%s/.%s.bak", dirname(copy_fname), basename(copy_fname)) ;
		while (access(backup_fname, F_OK) != -1) {
			strcpy(copy_fname, param->log_fname) ;
			sprintf(backup_fname, "%s/.%s.bak%d", dirname(copy_fname), basename(copy_fname), i) ;
			i++ ;
		}
		
		MPI_Barrier(MPI_COMM_WORLD) ;
		if (!param->rank) {
			fprintf(stderr, "Creating backup of log file %s -> %s\n", param->log_fname, backup_fname) ;
			sprintf(command, "cp -v %s %s", param->log_fname, backup_fname) ;
			system(command) ;
		}
	}
}

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
	params_from_config(config_fname, "emc", param) ;
	if (!continue_flag)
		backup_log_file(param) ;
#ifndef WITH_HDF5
	generate_output_dirs(param) ;
#endif // WITH_HDF5
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
	else if (param->recon_type == RECONRZ) {
		det_flag = -param->modes ;
		slice_gen = &slice_genrz ;
		slice_merge = &slice_mergerz ;
	}
	else {
		fprintf(stderr, "recon_type not recognized\n") ;
		return 1 ;
	}
	if ((qmax = detector_from_config(config_fname, "emc", &det, det_flag)) < 0.)
		return 1 ;
	if (param->radius > 0)
		remask_detector(det, param->radius) ;
	if (quat_from_config(config_fname, "emc", quat))
		return 1 ;
	divide_quat(param->rank, param->num_proc, param->modes, param->nonrot_modes, quat) ;
	if (data_from_config(config_fname, "emc", "in", det, frames))
		return 1 ;
	blacklist_from_config(config_fname, "emc", frames) ;
	if (iterate_from_config(config_fname, "emc", continue_flag, qmax, param, det, frames, iter))
		return 1 ;
	if (!param->rank && param->start_iter == 1)
		save_initial_iterate() ;

	gettimeofday(&t2, NULL) ;
	fprintf(stderr, "Completed setup: %f s\n", (double)(t2.tv_sec - t1.tv_sec) + 1.e-6*(t2.tv_usec - t1.tv_usec)) ;
	
	return 0 ;
}

void free_mem() {
	free_iterate(iter) ;
	iter = NULL ;
	free_data(param->need_scaling, frames) ;
	frames = NULL ;
	free_quat(quat) ;
	quat = NULL ;
	free_detector(det) ;
	det = NULL ;
	free(param) ;
	param = NULL ;
}

