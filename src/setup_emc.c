#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include "emc.h"

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

void generate_params(char *config_fname) {
	char line[1024], section_name[1024], *token ;
	
	param.known_scale = 0 ;
	param.start_iter = 1 ;
	param.beta_period = 100 ;
	param.beta_jump = 1. ;
	param.need_scaling = 0 ;
	param.alpha = 0. ;
	param.beta = 1. ;
	param.sigmasq = 0. ;
	strcpy(param.log_fname, "EMC.log") ;
	strcpy(param.output_folder, "data/") ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, config_section) == 0) {
			if (strcmp(token, "output_folder") == 0)
				strcpy(param.output_folder, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "log_file") == 0)
				strcpy(param.log_fname, strtok(NULL, " =\n")) ;
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
			else if (strcmp(token, "gaussian_sigma") == 0) {
				param.sigmasq = atof(strtok(NULL, " =\n")) ;
				param.sigmasq *= param.sigmasq ;
				fprintf(stderr, "sigma_squared = %f\n", param.sigmasq) ;
			}
		}
	}
	fclose(config_fp) ;
	if (!rank)
		fprintf(stderr, "Parsed params from config file\n") ;
}

void generate_output_dirs() {
	char line[1024] ;
	
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
}

void generate_blacklist(char *config_fname) {
	char blacklist_fname[1024] = {'\0'}, sel_string[1024] = {'\0'} ;
	char line[1024], section_name[1024], *token ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, config_section) == 0) {
			if (strcmp(token, "blacklist_file") == 0)
				strcpy(blacklist_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "selection") == 0)
				strcpy(sel_string, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	if (sel_string[0] == '\0') {
		make_blacklist(blacklist_fname, 0, frames) ;
	}
	else if (strcmp(sel_string, "odd_only") == 0) {
		if (!rank)
			fprintf(stderr, "Only processing 'odd' frames\n") ;
		make_blacklist(blacklist_fname, 1, frames) ;
	}
	else if (strcmp(sel_string, "even_only") == 0) {
		if (!rank)
			fprintf(stderr, "Only processing 'even' frames\n") ;
		make_blacklist(blacklist_fname, 2, frames) ;
	}
	else {
		fprintf(stderr, "Did not understand selection keyword: %s. Will process all frames\n", sel_string) ;
		make_blacklist(blacklist_fname, 0, frames) ;
	}
	
	if (!rank)
		fprintf(stderr, "%d/%d blacklisted frames\n", frames->num_blacklist, frames->tot_num_data) ;
}

int generate_iterate(char *config_fname, int continue_flag, double qmax) {
	FILE *fp ;
	char input_fname[1024] = {'\0'}, scale_fname[1024] = {'\0'} ;
	char line[1024], section_name[1024], *token ;
	iter->size = -1 ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, config_section) == 0) {
			if (strcmp(token, "size") == 0)
				iter->size = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "start_model_file") == 0)
				strcpy(input_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "scale_file") == 0)
				strcpy(scale_fname, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	generate_size(qmax, iter) ;
	
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
		parse_input(input_fname, frames[0].mean_count / det[0].rel_num_pix * 2., line, iter) ;
	}
	else {
		parse_input(input_fname, frames[0].mean_count / det[0].rel_num_pix * 2., NULL, iter) ;
	}
	
	return 0 ;
}

int setup(char *config_fname, int continue_flag) {
	FILE *fp ;
	double qmax = -1. ;
	strcpy(config_section, "emc") ;

	iter = malloc(sizeof(struct iterate)) ;
	quat = malloc(sizeof(struct rotation)) ;
	frames = malloc(sizeof(struct dataset)) ;
	merge_frames = NULL ;
	
	fp = fopen(config_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", config_fname) ;
		return 1 ;
	}
	fclose(fp) ;
	generate_params(config_fname) ;
	generate_output_dirs() ;
	if ((qmax = generate_detectors(config_fname, &det, 1)) < 0.)
		return 1 ;
	if (generate_quaternion(config_fname, quat))
		return 1 ;
	if (generate_data(config_fname, "in", det, frames))
		return 1 ;
	if (generate_data(config_fname, "merge", det, merge_frames))
		return 1 ;
	generate_blacklist(config_fname) ;
	if (generate_iterate(config_fname, continue_flag, qmax))
		return 1 ;
	
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

