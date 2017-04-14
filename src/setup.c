#include "emc.h"

int parse_det(char *fname) {
	int t, d ;
	double mean_pol = 0. ;
	
	rel_num_pix = 0 ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "det_fname %s not found. Exiting...1\n", fname) ;
		return 1 ;
	}
	fscanf(fp, "%d", &num_pix) ;
	det = malloc(4 * num_pix * sizeof(double)) ;
	mask = malloc(num_pix * sizeof(uint8_t)) ;
	for (t = 0 ; t < num_pix ; ++t) {
		for (d = 0 ; d < 4 ; ++d)
			fscanf(fp, "%lf", &det[t*4 + d]) ;
		fscanf(fp, "%" SCNu8, &mask[t]) ;
		if (mask[t] < 1)
			rel_num_pix++ ;
		mean_pol += det[t*4 + 3] ;
	}
	
	mean_pol /= num_pix ;
	for (t = 0 ; t < num_pix ; ++t)
		det[t*4 + 3] /= mean_pol ;
	
	fclose(fp) ;
	
	return 0 ;
}

int parse_quat(char *fname) {
	int r, t ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "quaternion file %s not found. Exiting.\n", fname) ;
		return 1 ;
	}
	double total_weight = 0. ;
	fscanf(fp, "%d", &num_rot) ;
	quat = malloc(num_rot * 5 * sizeof(double)) ;
	for (r = 0 ; r < num_rot ; ++r) {
		for (t = 0 ; t < 5 ; ++t)
			fscanf(fp, "%lf", &quat[r*5 + t]) ;
		total_weight += quat[r*5 + 4] ;
	}
	total_weight = 1. / total_weight ;
	for (r = 0 ; r < num_rot ; ++r)
		quat[r*5 + 4] *= total_weight ;
	fclose(fp) ;
	
	return 0 ;
}

int parse_dataset(char *fname, struct dataset *current) {
	int d ;
	current->ones_total = 0, current->multi_total = 0 ;
	
	strcpy(current->filename, fname) ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "data_fname %s not found. Exiting.\n", fname) ;
		return 1 ;
	}
	
	fread(&(current->num_data), sizeof(int), 1, fp) ;
	fread(&(current->num_pix), sizeof(int) , 1, fp) ;
	if (current->num_pix != num_pix)
		fprintf(stderr, "WARNING! The detector file and photons file %s do not"
		                "have the same number of pixels\n", current->filename) ;
	fseek(fp, 1016, SEEK_CUR) ;
	
	current->ones = malloc(current->num_data * sizeof(int)) ;
	current->multi = malloc(current->num_data * sizeof(int)) ;
	fread(current->ones, sizeof(int), current->num_data, fp) ;
	fread(current->multi, sizeof(int), current->num_data, fp) ;
	
	for (d = 0 ; d < current->num_data ; ++d) {
		current->ones_total += current->ones[d] ;
		current->multi_total += current->multi[d] ;
	}
	
	current->place_ones = malloc(current->ones_total * sizeof(int)) ;
	current->place_multi = malloc(current->multi_total * sizeof(int)) ;
	current->count_multi = malloc(current->multi_total * sizeof(int)) ;
	fread(current->place_ones, sizeof(int), current->ones_total, fp) ;
	fread(current->place_multi, sizeof(int), current->multi_total, fp) ;
	fread(current->count_multi, sizeof(int), current->multi_total, fp) ;
	
	fclose(fp) ;
	
	// Calculate mean count in the presence of mask
	long t, ones_counter = 0, multi_counter = 0 ;
	current->mean_count = 0. ;
	for (d = 0 ; d < current->num_data ; ++d) {
		for (t = 0 ; t < current->ones[d] ; ++t)
		if (mask[current->place_ones[ones_counter + t]] < 1)
			current->mean_count += 1. ;
		
		for (t = 0 ; t < current->multi[d] ; ++t)
		if (mask[current->place_multi[multi_counter + t]] < 1)
			current->mean_count += current->count_multi[multi_counter + t] ;
		
		ones_counter += current->ones[d] ;
		multi_counter += current->multi[d] ;
	}
	
	current->mean_count /= current->num_data ;
	
	return 0 ;
}

int parse_data(char *fname) {
	struct dataset *curr ;
	char data_fname[999] ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "data_flist %s not found. Exiting.\n", fname) ;
		return 1 ;
	}
	
	frames = malloc(sizeof(struct dataset)) ;
	frames->next = NULL ;
	
	if (fscanf(fp, "%s\n", data_fname) == 1) {
		if (parse_dataset(data_fname, frames))
			return 1 ;
	}
	else {
		fprintf(stderr, "No datasets found in %s\n", fname) ;
		return 1 ;
	}
	
	curr = frames ;
	tot_num_data = frames->num_data ;
	tot_mean_count = frames->num_data * frames->mean_count ;
	int num_datasets = 1 ;
	
	while (fscanf(fp, "%s\n", data_fname) == 1) {
		if (strlen(data_fname) == 0)
			continue ;
		curr->next = malloc(sizeof(struct dataset)) ;
		curr = curr->next ;
		curr->next = NULL ;
		
		if (parse_dataset(data_fname, curr))
			return 1 ;
		
		tot_num_data += curr->num_data ;
		tot_mean_count += curr->num_data * curr->mean_count ;
		num_datasets++ ;
	}
	fclose(fp) ;
	
	tot_mean_count /= tot_num_data ;
	
	return 0 ;
}

void parse_input(char *fname) {
	long x ;
	double model_mean ;
	
	model1 = malloc(size * size * size * sizeof(double)) ;
	model2 = malloc(size * size * size * sizeof(double)) ;
	inter_weight = malloc(size * size * size * sizeof(double)) ;
	
	srand(time(NULL)) ;
	
	if (rank == 0) {
		FILE *fp = fopen(fname, "r") ;
		if (fp == NULL) {
			if (!rank)
				fprintf(stderr, "Random start\n") ;
			
			model_mean = tot_mean_count / rel_num_pix * 2. ;
			for (x = 0 ; x < size * size * size ; ++x)
				model1[x] = ((double) rand() / RAND_MAX) * model_mean ;
		}
		else {
			if (!rank)
				fprintf(stderr, "Starting from %s\n", fname) ;
			
			fread(model1, sizeof(double), size * size * size, fp) ;
			fclose(fp) ;
		}
	}
	
	if (!rank && start_iter == 1) {
		char fname0[999] ;
		sprintf(fname0, "%s/output/intens_000.bin", output_folder) ;
		FILE *fp0 = fopen(fname0, "wb") ;
		fwrite(model1, sizeof(double), size*size*size, fp0) ;
		fclose(fp0) ;
	}
}

void calc_scale() {
	long d_counter = 0, ones_counter, multi_counter ;
	int d, t ;
	struct dataset *curr ;
	curr = frames ;
	FILE *fp ;
	char fname[999] ;
	
	scale = calloc(tot_num_data, sizeof(double)) ;
	count = calloc(tot_num_data, sizeof(int)) ;
	
	while (curr != NULL) {
		ones_counter = 0 ;
		multi_counter = 0 ;
		for (d = 0 ; d < curr->num_data ; ++d) {
			scale[d_counter + d] = 1. ;
			for (t = 0 ; t < curr->ones[d] ; ++t)
			if (mask[curr->place_ones[ones_counter + t]] < 1)
				count[d_counter + d]++ ;
			
			for (t = 0 ; t < curr->multi[d] ; ++t)
			if (mask[curr->place_multi[multi_counter + t]] < 1)
				count[d_counter + d] += curr->count_multi[multi_counter + t] ;
			
			ones_counter += curr->ones[d] ;
			multi_counter += curr->multi[d] ;
		}
		
		d_counter += curr->num_data ;
		curr = curr->next ;
	}
	
	if (!rank && start_iter == 1) {
		sprintf(fname, "%s/scale/scale_000.dat", output_folder) ;
		fp = fopen(fname, "w") ;
		for (d = 0 ; d < tot_num_data ; ++d)
			fprintf(fp, "%.6e\n", scale[d]) ;
		fclose(fp) ;
	}
}

void calc_sum_fact() {
	int d, t, d_counter = 0 ;
	long multi_counter ;
	struct dataset *curr = frames ;
	
	sum_fact = calloc(tot_num_data, sizeof(double)) ;
	
	while (curr != NULL) {
		multi_counter = 0 ;
		
		for (d = d_counter ; d < d_counter + curr->num_data ; ++d) {
			for (t = 0 ; t < curr->multi[d - d_counter] ; ++t)
			if (mask[curr->place_multi[multi_counter + t]] < 1)
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
	blacklist = calloc(tot_num_data, sizeof(uint8_t)) ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp != NULL) {
		if (!rank)
			fprintf(stderr, "Blacklisting frames according to %s\n", fname) ;
		for (d = 0 ; d < tot_num_data ; ++d) {
			fscanf(fp, "%" SCNu8 "\n", &blacklist[d]) ;
			if (blacklist[d])
				num_blacklist++ ;
		}
		fclose(fp) ;
	}
	
	if (flag > 0) {
		for (d = 0 ; d < tot_num_data ; ++d)
		if (!blacklist[d]) {
			blacklist[d] = current ;
			num_blacklist += current ;
			current = 1 - current ;
		}
	}
	
	if (!rank)
		fprintf(stderr, "%d/%d blacklisted frames\n", num_blacklist, tot_num_data) ;
}

void parse_scale(char *fname) {
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		if (!rank)
			fprintf(stderr, "Using uniform scale factors\n") ;
	}
	else {
		if (!rank)
			fprintf(stderr, "Using scale factors from %s\n", fname) ;
		known_scale = 1 ;
		int d ;
		for (d = 0 ; d < tot_num_data ; ++d)
			fscanf(fp, "%lf", &scale[d]) ;
		fclose(fp) ;
	}
}

int setup(char *config_fname, int continue_flag) {
	FILE *fp ;
	char det_fname[999], quat_fname[999] ;
	char data_flist[999], input_fname[999] ;
	char scale_fname[999], blacklist_fname[999] ;
	char data_fname[999], out_data_fname[999] ;
	char merge_flist[999], merge_fname[999] ;
	char out_det_fname[999], sel_string[999] ;
	int good_section ;
	double qmax, qmin, detd, pixsize, ewald_rad ;
	int dets_x, dets_y, detsize, num_div ;
	
	good_section = 0 ;
	known_scale = 0 ;
	start_iter = 1 ;
	strcpy(log_fname, "EMC.log") ;
	strcpy(output_folder, "data/") ;
	data_flist[0] = '\0' ;
	data_fname[0] = '\0' ;
	merge_flist[0] = '\0' ;
	merge_fname[0] = '\0' ;
	quat_fname[0] = '\0' ;
	sel_string[0] = '\0' ;
	merge_frames = NULL ;
	detd = 0. ;
	pixsize = 0. ;
	detsize = 0 ;
	dets_x = 0 ;
	dets_y = 0 ;
	size = -1 ;
	beta_period = 100 ;
	beta_jump = 1. ;
	num_div = -1 ;
	need_scaling = 0 ;
	alpha = 0. ;
	beta = 1. ;
	ewald_rad = -1. ;
	icosahedral_flag = 0 ;
	
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
			need_scaling = atoi(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "alpha") == 0)
			alpha = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "beta") == 0)
			beta = atof(strtok(NULL, " =\n")) ;
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
			strcpy(output_folder, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "log_file") == 0)
			strcpy(log_fname, strtok(NULL, " =\n")) ;
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
			beta_jump = atof(strtok(NULL, " =\n")) ;
			beta_period = atoi(strtok(NULL, " =\n")) ;
		}
		else if (strcmp(token, "selection") == 0)
			strcpy(sel_string, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "sym_icosahedral") == 0)
			icosahedral_flag = atoi(strtok(NULL, " =\n")) ;
	}
	fclose(fp) ;

	if (strcmp(det_fname, "make_detector:::out_detector_file") == 0)
		strcpy(det_fname, out_det_fname) ;
	if (strcmp(data_fname, "make_data:::out_photons_file") == 0)
		strcpy(data_fname, out_data_fname) ;
	
	if (detsize == 0 || pixsize == 0. || detd == 0.) {
		fprintf(stderr, "Need detector parameters, detd, detsize, pixsize\n") ;
		return 1 ;
	}
	
	sprintf(line, "%s/output", output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/weights", output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/mutualInfo", output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/scale", output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/orientations", output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/likelihood", output_folder) ;
	mkdir(line, 0750) ;
	
	double hx = (dets_x - 1) / 2 * pixsize ;
	double hy = (dets_y - 1) / 2 * pixsize ;
	qmax = 2. * sin(0.5 * atan(sqrt(hx*hx + hy*hy)/detd)) ;
	qmin = 2. * sin(0.5 * atan(pixsize/detd)) ;
	if (ewald_rad == -1.)
		size = 2 * ceil(qmax / qmin) + 3 ;
	else
		size = 2 * ceil(qmax / qmin * ewald_rad * pixsize / detd) + 3 ;
	center = size / 2 ;
	fprintf(stderr, "Generating 3D volume of size %d\n", size) ;
	
	if (parse_det(det_fname))
		return 1 ;

	if (num_div > 0 && quat_fname[0] != '\0') {
		fprintf(stderr, "Config file contains both num_div as well as in_quat_file. Pick one.\n") ;
		return 1 ;
	}
	else if (num_div > 0)
		num_rot = quat_gen(num_div, &quat, icosahedral_flag) ;
	else if (parse_quat(quat_fname))
			return 1 ;
	
	num_rot_p = num_rot / num_proc ;
	if (rank < (num_rot % num_proc))
		num_rot_p++ ;
	if (num_proc > 1) {
		char hname[99] ;
		gethostname(hname, 99) ;
		fprintf(stderr, "%d: %s: num_rot_p = %d\n", rank, hname, num_rot_p) ;
	}
	
	if (data_flist[0] != '\0' && data_fname[0] != '\0') {
		fprintf(stderr, "Config file contains both in_photons_file and in_photons_list. Pick one.\n") ;
		return 1 ;
	}
	else if (data_flist[0] == '\0') {
		frames = malloc(sizeof(struct dataset)) ;
		frames->next = NULL ;
		if (parse_dataset(data_fname, frames))
			return 1 ;
		tot_num_data = frames->num_data ;
		tot_mean_count = frames->mean_count ;
	}
	else if (parse_data(data_flist))
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
		if (parse_dataset(merge_fname, merge_frames))
			return 1 ;
	}
	else if (merge_flist[0] != '\0') {
		if (parse_data(merge_flist))
			return 1 ;
	}
	
	calc_sum_fact() ;
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
	
	if (continue_flag) {
		fp = fopen(log_fname, "r") ;
		if (fp == NULL) {
			fprintf(stderr, "No log file found to continue run\n") ;
			continue_flag = 0 ;
		}
		else {
			while (!feof(fp))
				fgets(line, 500, fp) ;
			sscanf(line, "%d", &start_iter) ;
			fclose(fp) ;
			
			sprintf(input_fname, "%s/output/intens_%.3d.bin", output_folder, start_iter) ;
			if (need_scaling)
				sprintf(scale_fname, "%s/scale/scale_%.3d.dat", output_folder, start_iter) ;
			start_iter += 1 ;
			if (!rank)
				fprintf(stderr, "Continuing from previous run starting from iteration %d.\n", start_iter) ;
		}
	}
	
	if (need_scaling) {
		calc_scale() ;
		parse_scale(scale_fname) ;
	}
	
	parse_input(input_fname) ;
	
	return 0 ;
}

void free_mem() {
	free(det) ;
	free(quat) ;
//	free(model1) ;
	free(model2) ;
	free(inter_weight) ;
	if (need_scaling) {
		free(scale) ;
		free(count) ;
	}
	
	free(frames->ones) ;
	free(frames->multi) ;
	free(frames->place_ones) ;
	free(frames->place_multi) ;
	free(frames->count_multi) ;
}

