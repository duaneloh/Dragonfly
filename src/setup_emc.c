#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include "emc.h"

int setup(char *config_fname, int continue_flag) {
	FILE *fp ;
	char det_fname[1024], quat_fname[1024] ;
	char data_flist[1024], input_fname[1024] ;
	char scale_fname[1024], blacklist_fname[1024] ;
	char data_fname[1024], out_data_fname[1024] ;
	char merge_flist[1024], merge_fname[1024] ;
	char out_det_fname[1024], det_flist[1024] ; 
	char section_name[1024], sel_string[1024] ;
	int num, sym_icosahedral = 0 ;
	double qmax = -1. ;
	int num_div = -1 ;

	// Set default values of
	// 	... local variables
	data_flist[0] = '\0' ;
	data_fname[0] = '\0' ;
	det_flist[0] = '\0' ;
	det_fname[0] = '\0' ;
	merge_flist[0] = '\0' ;
	merge_fname[0] = '\0' ;
	quat_fname[0] = '\0' ;
	sel_string[0] = '\0' ;
	//	... structured variables
	param.known_scale = 0 ;
	param.start_iter = 1 ;
	strcpy(param.log_fname, "EMC.log") ;
	strcpy(param.output_folder, "data/") ;
	param.beta_period = 100 ;
	param.beta_jump = 1. ;
	param.need_scaling = 0 ;
	param.alpha = 0. ;
	param.beta = 1. ;
	param.sigmasq = 0. ;
	iter = malloc(sizeof(struct iterate)) ;
	quat = malloc(sizeof(struct rotation)) ;
	frames = malloc(sizeof(struct dataset)) ;
	merge_frames = NULL ;
	iter->size = -1 ;

	// Parse config file options
	char line[1024], *token ;
	fp = fopen(config_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", config_fname) ;
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
		}
		
		if (strcmp(section_name, "make_detector") == 0) {
			if (strcmp(token, "out_detector_file") == 0)
				strcpy(out_det_fname, strtok(NULL, " =\n")) ;
		}
		else if (strcmp(section_name, "make_data") == 0) {
			if (strcmp(token, "out_photons_file") == 0)
				strcpy(out_data_fname, strtok(NULL, " =\n")) ;
		}
		else if (strcmp(section_name, "emc") == 0) {
			if (strcmp(token, "in_photons_file") == 0)
				strcpy(data_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_photons_list") == 0)
				strcpy(data_flist, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "merge_photons_file") == 0)
				strcpy(merge_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "merge_photons_list") == 0)
				strcpy(merge_flist, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_detector_file") == 0)
				strcpy(det_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_detector_list") == 0)
				strcpy(det_flist, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "output_folder") == 0)
				strcpy(param.output_folder, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "log_file") == 0)
				strcpy(param.log_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "start_model_file") == 0)
				strcpy(input_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "num_div") == 0)
				num_div = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "size") == 0)
				iter->size = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_quat_file") == 0)
				strcpy(quat_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "blacklist_file") == 0)
				strcpy(blacklist_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "scale_file") == 0)
				strcpy(scale_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "need_scaling") == 0)
				param.need_scaling = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "alpha") == 0)
				param.alpha = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "beta") == 0)
				param.beta = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "beta_schedule") == 0) {
				param.beta_jump = atof(strtok(NULL, " =\n")) ;
				param.beta_period = atoi(strtok(NULL, " =\n")) ;
			}
			else if (strcmp(token, "selection") == 0)
				strcpy(sel_string, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "sym_icosahedral") == 0)
				sym_icosahedral = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "gaussian_sigma") == 0) {
				param.sigmasq = atof(strtok(NULL, " =\n")) ;
				param.sigmasq *= param.sigmasq ;
				fprintf(stderr, "sigma_squared = %f\n", param.sigmasq) ;
			}
		}
	}
	fclose(fp) ;
	fprintf(stderr, "Parsed config file %s\n", config_fname) ;

	// Check for referenced arguments
	if (strcmp(det_fname, "make_detector:::out_detector_file") == 0)
		strcpy(det_fname, out_det_fname) ;
	if (strcmp(data_fname, "make_data:::out_photons_file") == 0)
		strcpy(data_fname, out_data_fname) ;

	// Create output subdirectories
	sprintf(line, "%s/output", param.output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/weights", param.output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/mutualInfo", param.output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/scale", param.output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/orientations", param.output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/likelihood", param.output_folder) ;
	mkdir(line, 0750) ;

	// Generate detector(s)
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
	if (!rank) {
		fprintf(stderr, "Number of unique detectors = %d\n", det[0].num_det) ;
		fprintf(stderr, "Number of detector files = %d\n", det[0].num_dfiles) ;
	}

	// Calculate size and center
	if (iter->size < 0) {
		iter->size = 2*ceil(qmax) + 3 ;
		fprintf(stderr, "Calculated 3D volume size = %ld\n", iter->size) ;
	}
	else {
		fprintf(stderr, "Provided 3D volume size = %ld\n", iter->size) ;
	}
	iter->center = iter->size / 2 ;

	// Generate quaternions
	quat->icosahedral_flag = sym_icosahedral ;
	if (num_div > 0 && quat_fname[0] != '\0') {
		fprintf(stderr, "Config file contains both num_div as well as in_quat_file. Pick one.\n") ;
		return 1 ;
	}
	else if (num_div > 0)
		num = quat_gen(num_div, quat) ;
	else
		num = parse_quat(quat_fname, quat) ;
	if (num < 0)
		return 1 ;
	
	divide_quat(rank, num_proc, quat) ;

	// Generate data
	int num_datasets ;
	frames->next = NULL ;
	if (data_flist[0] != '\0' && data_fname[0] != '\0') {
		fprintf(stderr, "Config file contains both in_photons_file and in_photons_list. Pick one.\n") ;
		return 1 ;
	}
	else if (data_fname[0] != '\0') {
		if (parse_dataset(data_fname, det, frames))
			return 1 ;
		frames->num_data_prev = 0 ;
		calc_sum_fact(det, frames) ;
		num_datasets = 1 ;
	}
	else if (data_flist[0] != '\0') {
		if ((num_datasets = parse_data(data_flist, det, frames)) < 0)
			return 1 ;
	}
	else {
		fprintf(stderr, "Need either in_photons_file or in_photons_list.\n") ;
		return 1 ;
	}
	if (det[0].num_dfiles > 0 && det[0].num_dfiles != num_datasets) {
		fprintf(stderr, "Number of detector files and emc files don't match (%d vs %d)\n", det[0].num_dfiles, num_datasets) ;
		return 1 ;
	}
	if (!rank)
		fprintf(stderr, "Number of dataset files = %d\n", num_datasets) ;
	
	if (merge_flist[0] != '\0' && merge_fname[0] != '\0') {
		fprintf(stderr, "Config file contains both merge_photons_file and merge_photons_list. Pick one.\n") ;
		return 1 ;
	}
	else if (merge_fname[0] != '\0') {
		if (!rank)
			fprintf(stderr, "Parsing merge file %s\n", merge_fname) ;
		merge_frames = malloc(sizeof(struct dataset)) ;
		merge_frames->next = NULL ;
		merge_frames->num_data_prev = 0 ;
		if (parse_dataset(merge_fname, det, merge_frames))
			return 1 ;
	}
	else if (merge_flist[0] != '\0') {
		merge_frames = malloc(sizeof(struct dataset)) ;
		merge_frames->next = NULL ;
		if (parse_data(merge_flist, det, merge_frames))
			return 1 ;
	}

	// Generate blacklist
	if (sel_string[0] == '\0') {
		gen_blacklist(blacklist_fname, 0, frames) ;
	}
	else if (strcmp(sel_string, "odd_only") == 0) {
		if (!rank)
			fprintf(stderr, "Only processing 'odd' frames\n") ;
		gen_blacklist(blacklist_fname, 1, frames) ;
	}
	else if (strcmp(sel_string, "even_only") == 0) {
		if (!rank)
			fprintf(stderr, "Only processing 'even' frames\n") ;
		gen_blacklist(blacklist_fname, 2, frames) ;
	}
	else {
		fprintf(stderr, "Did not understand selection keyword: %s. Will process all frames\n", sel_string) ;
		gen_blacklist(blacklist_fname, 0, frames) ;
	}
	
	if (!rank)
		fprintf(stderr, "%d/%d blacklisted frames\n", frames->num_blacklist, frames->tot_num_data) ;

	// Generate iterate
	if (continue_flag) {
		fp = fopen(param.log_fname, "r") ;
		if (fp == NULL) {
			fprintf(stderr, "No log file found to continue run\n") ;
			return 1 ;
		}
		else {
			while (!feof(fp))
				fgets(line, 500, fp) ;
			sscanf(line, "%d", &param.start_iter) ;
			fclose(fp) ;
			
			sprintf(input_fname, "%s/output/intens_%.3d.bin", param.output_folder, param.start_iter) ;
			if (param.need_scaling)
				sprintf(scale_fname, "%s/scale/scale_%.3d.dat", param.output_folder, param.start_iter) ;
			param.start_iter += 1 ;
			if (!rank)
				fprintf(stderr, "Continuing from previous run starting from iteration %d.\n", param.start_iter) ;
		}
	}
	
	if (param.need_scaling) {
		if (!rank && param.start_iter == 1) {
			sprintf(line, "%s/scale/scale_000.dat", param.output_folder) ;
			calc_scale(frames, det, line, iter) ;
		}
		else {
			calc_scale(frames, det, NULL, iter) ;
		}
		param.known_scale = parse_scale(scale_fname, frames, iter) ;
	}
	
	if (!rank && param.start_iter == 1) {
		sprintf(line, "%s/output/intens_000.bin", param.output_folder) ;
		//parse_input(input_fname, frames->tot_mean_count / det->rel_num_pix * 2., line, iter) ;
		parse_input(input_fname, frames[0].mean_count / det[0].rel_num_pix * 2., line, iter) ;
	}
	else {
		//parse_input(input_fname, frames->tot_mean_count / det->rel_num_pix * 2., NULL, iter) ;
		parse_input(input_fname, frames[0].mean_count / det[0].rel_num_pix * 2., NULL, iter) ;
	}
	
	return 0 ;
}

void free_mem() {
	free_iterate(param.need_scaling, iter) ;
	free(iter) ;
	if (merge_frames != NULL) {
		free_data(param.need_scaling, merge_frames) ;
		free(merge_frames) ;
	}
	free_data(param.need_scaling, frames) ;
	free(frames) ;
	free_quat(quat) ;
	free(quat) ;
	free_detector(det) ;
	free(det) ;
}

