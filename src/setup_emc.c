#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <inttypes.h>
#include <gsl/gsl_sf_gamma.h>
#include "emc.h"

void calc_sum_fact() {
	int d, t, d_counter = 0 ;
	long multi_counter ;
	struct dataset *curr = frames ;
	
	sum_fact = calloc(frames->tot_num_data, sizeof(double)) ;
	
	while (curr != NULL) {
		multi_counter = 0 ;
		
		for (d = d_counter ; d < d_counter + curr->num_data ; ++d) {
			for (t = 0 ; t < curr->multi[d - d_counter] ; ++t)
			if (det->mask[curr->place_multi[multi_counter + t]] < 1)
				sum_fact[d] += gsl_sf_lnfact(curr->count_multi[multi_counter + t]) ;
			multi_counter += curr->multi[d - d_counter] ;
		}
		
		d_counter += curr->num_data ;
		curr = curr->next ;
	}
}

void gen_blacklist(char *fname, int flag) {
	int d, current = flag%2 ;
	num_blacklist = 0 ;
	blacklist = calloc(frames->tot_num_data, sizeof(uint8_t)) ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp != NULL) {
		if (!rank)
			fprintf(stderr, "Blacklisting frames according to %s\n", fname) ;
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			fscanf(fp, "%" SCNu8 "\n", &blacklist[d]) ;
			if (blacklist[d])
				num_blacklist++ ;
		}
		fclose(fp) ;
	}
	
	if (flag > 0) {
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (!blacklist[d]) {
			blacklist[d] = current ;
			num_blacklist += current ;
			current = 1 - current ;
		}
	}
	
	if (!rank)
		fprintf(stderr, "%d/%d blacklisted frames\n", num_blacklist, frames->tot_num_data) ;
}

int setup(char *config_fname, int continue_flag) {
	FILE *fp ;
	char det_fname[999], quat_fname[999] ;
	char data_flist[999], input_fname[999] ;
	char scale_fname[999], blacklist_fname[999] ;
	char data_fname[999], out_data_fname[999] ;
	char merge_flist[999], merge_fname[999] ;
	char out_det_fname[999], sel_string[999] ;
	int num, good_section = 0, sym_icosahedral = 0 ;
	double qmax, qmin, detd = 0., pixsize = 0., ewald_rad = -1. ;
	int dets_x = 0, dets_y = 0, detsize = 0, num_div = -1 ;

	// Set default values of
	// 	... local variables
	data_flist[0] = '\0' ;
	data_fname[0] = '\0' ;
	merge_flist[0] = '\0' ;
	merge_fname[0] = '\0' ;
	quat_fname[0] = '\0' ;
	sel_string[0] = '\0' ;
	//	... structured variables
	param.known_scale = 0 ;
	param.start_iter = 1 ;
	strcpy(param.log_fname, "EMC.log") ;
	strcpy(param.output_folder, "data/") ;
	merge_frames = NULL ;
	param.beta_period = 100 ;
	param.beta_jump = 1. ;
	param.need_scaling = 0 ;
	param.alpha = 0. ;
	param.beta = 1. ;
	iter->size = -1 ;

	// Parse config file options
	char line[999], *token ;
	fp = fopen(config_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", config_fname) ;
		return 1 ;
	}
	while (fgets(line, 999, fp) != NULL) {
		token = strtok(line, " =") ;
		if (token[0] == '#' || token[0] == '\n') {
			continue ;
		}
		else if (token[0] == '[') {
			token = strtok(token, "[]") ;
			if (strcmp(token, "emc") == 0 ||
			    strcmp(token, "parameters") == 0 ||
			    strcmp(token, "make_detector") == 0 ||
			    strcmp(token, "make_data") == 0)
				good_section = 1 ;
			else
				good_section = 0 ;
			continue ;
		}
		if (!good_section)
			continue ;
		
		// [parameters]
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
		else if (strcmp(token, "need_scaling") == 0)
			param.need_scaling = atoi(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "alpha") == 0)
			param.alpha = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "beta") == 0)
			param.beta = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "ewald_rad") == 0)
			ewald_rad = atof(strtok(NULL, " =\n")) ;
		// [make_detector]
		else if (strcmp(token, "out_detector_file") == 0)
			strcpy(out_det_fname, strtok(NULL, " =\n")) ;
		// [make_data]
		else if (strcmp(token, "out_photons_file") == 0)
			strcpy(out_data_fname, strtok(NULL, " =\n")) ;
		// [emc]
		else if (strcmp(token, "in_photons_file") == 0)
			strcpy(data_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "in_photons_file") == 0)
			strcpy(data_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "in_photons_list") == 0)
			strcpy(data_flist, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "merge_photons_file") == 0)
			strcpy(merge_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "merge_photons_list") == 0)
			strcpy(merge_flist, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "output_folder") == 0)
			strcpy(param.output_folder, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "log_file") == 0)
			strcpy(param.log_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "start_model_file") == 0)
			strcpy(input_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "in_detector_file") == 0)
			strcpy(det_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "num_div") == 0)
			num_div = atoi(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "in_quat_file") == 0)
			strcpy(quat_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "blacklist_file") == 0)
			strcpy(blacklist_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "scale_file") == 0)
			strcpy(scale_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "beta_schedule") == 0) {
			param.beta_jump = atof(strtok(NULL, " =\n")) ;
			param.beta_period = atoi(strtok(NULL, " =\n")) ;
		}
		else if (strcmp(token, "selection") == 0)
			strcpy(sel_string, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "sym_icosahedral") == 0)
			sym_icosahedral = atoi(strtok(NULL, " =\n")) ;
	}
	fclose(fp) ;

	// Check for referenced arguments
	if (strcmp(det_fname, "make_detector:::out_detector_file") == 0)
		strcpy(det_fname, out_det_fname) ;
	if (strcmp(data_fname, "make_data:::out_photons_file") == 0)
		strcpy(data_fname, out_data_fname) ;

	// Check for compulsory parameter options
	if (detsize == 0 || pixsize == 0. || detd == 0.) {
		fprintf(stderr, "Need detector parameters, detd, detsize, pixsize\n") ;
		return 1 ;
	}

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

	// Calculate size and center
	double hx = (dets_x - 1) / 2 * pixsize ;
	double hy = (dets_y - 1) / 2 * pixsize ;
	qmax = 2. * sin(0.5 * atan(sqrt(hx*hx + hy*hy)/detd)) ;
	qmin = 2. * sin(0.5 * atan(pixsize/detd)) ;
	if (ewald_rad == -1.)
		iter->size = 2 * ceil(qmax / qmin) + 3 ;
	else
		iter->size = 2 * ceil(qmax / qmin * ewald_rad * pixsize / detd) + 3 ;
	iter->center = iter->size / 2 ;
	fprintf(stderr, "Generating 3D volume of size %ld\n", iter->size) ;

	// Generate detector
	det = malloc(sizeof(struct detector)) ;
	if (parse_detector(det_fname, det))
		return 1 ;

	// Generate quaternions
	quat = malloc(sizeof(struct rotation)) ;
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
	
	if (merge_flist[0] != '\0' && merge_fname[0] != '\0') {
		fprintf(stderr, "Config file contains both merge_photons_file and merge_photons_list. Pick one.\n") ;
		return 1 ;
	}
	else if (merge_fname[0] != '\0') {
		if (!rank)
			fprintf(stderr, "Parsing merge file %s\n", merge_fname) ;
		merge_frames = malloc(sizeof(struct dataset)) ;
		merge_frames->next = NULL ;
		if (parse_dataset(merge_fname, det, merge_frames))
			return 1 ;
	}
	else if (merge_flist[0] != '\0') {
		merge_frames = malloc(sizeof(struct dataset)) ;
		merge_frames->next = NULL ;
		if (parse_data(merge_flist, det, merge_frames))
			return 1 ;
	}
	
	calc_sum_fact() ;

	// Generate blacklist
	if (sel_string[0] == '\0')
		gen_blacklist(blacklist_fname, 0) ;
	else if (strcmp(sel_string, "odd_only") == 0) {
		if (!rank)
			fprintf(stderr, "Only processing 'odd' frames\n") ;
		gen_blacklist(blacklist_fname, 1) ;
	}
	else if (strcmp(sel_string, "even_only") == 0) {
		if (!rank)
			fprintf(stderr, "Only processing 'even' frames\n") ;
		gen_blacklist(blacklist_fname, 2) ;
	}
	else {
		fprintf(stderr, "Did not understand selection keyword: %s. Will process all frames\n", sel_string) ;
		gen_blacklist(blacklist_fname, 0) ;
	}

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
		parse_scale(scale_fname, frames, iter) ;
	}
	
	if (!rank && param.start_iter == 1) {
		sprintf(line, "%s/output/intens_000.bin", param.output_folder) ;
		parse_input(input_fname, frames->tot_mean_count / det->rel_num_pix * 2., line, iter) ;
	}
	else {
		parse_input(input_fname, frames->tot_mean_count / det->rel_num_pix * 2., NULL, iter) ;
	}
	
	return 0 ;
}

void free_mem() {
	free_iterate(param.need_scaling, iter) ;
	free_data(param.need_scaling, frames) ;
	if (merge_frames != NULL)
		free_data(param.need_scaling, merge_frames) ;
	free_quat(quat) ;
	free_detector(det) ;
}

