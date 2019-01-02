#include "dataset.h"

static char *generate_token(char *line, char *section_name) {
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

static void absolute_strcpy(char *config_folder, char *path, char *rel_path) {
	if (rel_path[0] == '/' || strstr(rel_path, ":::") != NULL) {
		strcpy(path, rel_path) ;
	}
	else {
		strcpy(&path[strlen(config_folder)], rel_path) ;
		strncpy(path, config_folder, strlen(config_folder)) ;
	}
}

int generate_data(char *config_fname, char *config_section, char *type_string, struct detector *det_list, struct dataset *frames_list) {
	int num_datasets = 0 ;
	char data_fname[1024] = {'\0'}, data_flist[1024] = {'\0'}, out_data_fname[1024] = {'\0'} ;
	char fname_opt[64], flist_opt[64] ;
	char line[1024], section_name[1024], config_folder[1024], *token ;
	char *temp_fname = strndup(config_fname, 1024) ;
	sprintf(config_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	sprintf(fname_opt, "%s_photons_file", type_string) ;
	sprintf(flist_opt, "%s_photons_list", type_string) ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "make_data") == 0) {
			if (strcmp(token, "out_photons_file") == 0)
				absolute_strcpy(config_folder, out_data_fname, strtok(NULL, " =\n")) ;
		}
		else if (strcmp(section_name, config_section) == 0) {
			if (strcmp(token, fname_opt) == 0)
				absolute_strcpy(config_folder, data_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, flist_opt) == 0)
				absolute_strcpy(config_folder, data_flist, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	if (strcmp(data_fname, "make_data:::out_photons_file") == 0)
		strcpy(data_fname, out_data_fname) ;
	
	if (data_flist[0] != '\0' && data_fname[0] != '\0') {
		fprintf(stderr, "Config file contains both in_photons_file and in_photons_list. Pick one.\n") ;
		return 1 ;
	}
	else if (data_fname[0] != '\0') {
		if (frames_list == NULL)
			frames_list = malloc(sizeof(struct dataset)) ;
		if (parse_dataset(data_fname, det_list, frames_list))
			return 1 ;
		frames_list->num_data_prev = 0 ;
		frames_list->next = NULL ;
		calc_sum_fact(det_list, frames_list) ;
		num_datasets = 1 ;
	}
	else if (data_flist[0] != '\0') {
		if (frames_list == NULL)
			frames_list = malloc(sizeof(struct dataset)) ;
		frames_list->next = NULL ;
		if ((num_datasets = parse_data(data_flist, det_list, frames_list)) < 0)
			return 1 ;
	}
	else if (strcmp(type_string, "in") == 0) {
		fprintf(stderr, "Need either in_photons_file or in_photons_list.\n") ;
		return 1 ;
	}
	
	if (strcmp(type_string, "in") == 0) {
		if (det_list[0].num_dfiles > 0 && det_list[0].num_dfiles != num_datasets) {
			fprintf(stderr, "Number of detector files and photon files don't match (%d vs %d)\n", det_list[0].num_dfiles, num_datasets) ;
			return 1 ;
		}
		fprintf(stderr, "Number of dataset files = %d\n", num_datasets) ;
	}
	
	return 0 ;
}

void calc_sum_fact(struct detector *det, struct dataset *frames) {
	int d, t ;
	struct dataset *curr = frames ;
	
	frames->sum_fact = calloc(frames->tot_num_data, sizeof(double)) ;
	
	while (curr != NULL) {
		if (curr->type == 0) {
			for (d = 0 ; d < curr->num_data ; ++d)
			for (t = 0 ; t < curr->multi[d] ; ++t)
			if (det->mask[curr->place_multi[curr->multi_accum[d] + t]] < 1)
				frames->sum_fact[curr->num_data_prev+d] += gsl_sf_lnfact(curr->count_multi[curr->multi_accum[d] + t]) ;
		}
		else if (curr->type == 1) {
			for (d = 0 ; d < curr->num_data ; ++d)
			for (t = 0 ; t < curr->num_pix ; ++t)
			if (det->mask[t] < 1)
				frames->sum_fact[curr->num_data_prev+d] += gsl_sf_lnfact(curr->int_frames[d*curr->num_pix + t]) ;
		}
		else if (curr->type == 2) {
			for (d = 0 ; d < curr->num_data ; ++d)
				frames->sum_fact[curr->num_data_prev+d] = 0. ;
		}
			
		curr = curr->next ;
	}
}

int parse_dataset(char *fname, struct detector *det, struct dataset *current) {
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
	current->tot_num_data = current->num_data ;
	if (current->num_pix != det->num_pix)
		fprintf(stderr, "WARNING! The detector file and photons file %s do not "
		                "have the same number of pixels\n", current->filename) ;
	fread(&(current->type), sizeof(int), 1, fp) ;
	fseek(fp, 1024, SEEK_SET) ;
	if (current->type == 0) { // Sparse
		current->ones = malloc(current->num_data * sizeof(int)) ;
		current->multi = malloc(current->num_data * sizeof(int)) ;
		fread(current->ones, sizeof(int), current->num_data, fp) ;
		fread(current->multi, sizeof(int), current->num_data, fp) ;
		
		current->ones_accum = malloc(current->num_data * sizeof(long)) ;
		current->multi_accum = malloc(current->num_data * sizeof(long)) ;
		current->ones_accum[0] = 0 ;
		current->multi_accum[0] = 0 ;
		for (d = 1 ; d < current->num_data ; ++d) {
			current->ones_accum[d] = current->ones_accum[d-1] + current->ones[d-1] ;
			current->multi_accum[d] = current->multi_accum[d-1] + current->multi[d-1] ;
		}
		current->ones_total = current->ones_accum[current->num_data-1] + current->ones[current->num_data-1] ;
		current->multi_total = current->multi_accum[current->num_data-1] + current->multi[current->num_data-1] ;
		
		current->place_ones = malloc(current->ones_total * sizeof(int)) ;
		current->place_multi = malloc(current->multi_total * sizeof(int)) ;
		current->count_multi = malloc(current->multi_total * sizeof(int)) ;
		fread(current->place_ones, sizeof(int), current->ones_total, fp) ;
		fread(current->place_multi, sizeof(int), current->multi_total, fp) ;
		fread(current->count_multi, sizeof(int), current->multi_total, fp) ;
	}
	else if (current->type == 1) { // Dense integer
		fprintf(stderr, "%s is a dense integer emc file\n", current->filename) ;
		current->int_frames = malloc(current->num_pix * current->num_data * sizeof(int)) ;
		fread(current->int_frames, sizeof(int), current->num_pix * current->num_data, fp) ;
	}
	else if (current->type == 2) { // Dense double
		fprintf(stderr, "%s is a dense double precision emc file\n", current->filename) ;
		current->frames = malloc(current->num_pix * current->num_data * sizeof(double)) ;
		fread(current->frames, sizeof(double), current->num_pix * current->num_data, fp) ;
	}
	else {
		fprintf(stderr, "Unknown dataset type %d\n", current->type) ;
		fclose(fp) ;
		return 1 ;
	}
	fclose(fp) ;

	// Calculate mean count in the presence of mask
	long t ;
	current->mean_count = 0. ;
	for (d = 0 ; d < current->num_data ; ++d) {
		if (current->type == 0) {
			for (t = 0 ; t < current->ones[d] ; ++t)
			if (det->mask[current->place_ones[current->ones_accum[d] + t]] < 1)
				current->mean_count += 1. ;
			
			for (t = 0 ; t < current->multi[d] ; ++t)
			if (det->mask[current->place_multi[current->multi_accum[d] + t]] < 1)
				current->mean_count += current->count_multi[current->multi_accum[d] + t] ;
		}
		else if (current->type == 1) {
			for (t = 0 ; t < current->num_pix ; ++t)
			if (det->mask[t] < 1)
				current->mean_count += current->int_frames[d*current->num_pix + t] ;
		}
		else if (current->type == 2) {
			for (t = 0 ; t < current->num_pix ; ++t)
			if (det->mask[t] < 1)
				current->mean_count += current->frames[d*current->num_pix + t] ;
		}
	}
	
	current->mean_count /= current->num_data ;
	current->tot_mean_count = current->mean_count ;
	current->blacklist = NULL ;
	current->sum_fact = NULL ;
	
	return 0 ;
}

int parse_data(char *flist, struct detector *det, struct dataset *frames) {
	struct dataset *curr ;
	char data_fname[1024] ;
	char flist_folder[1024], rel_fname[1024] ;
	char *temp_fname = strndup(flist, 1024) ;
	sprintf(flist_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	
	FILE *fp = fopen(flist, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "data_flist %s not found. Exiting.\n", flist) ;
		return -1 ;
	}
	
	if (fscanf(fp, "%s\n", rel_fname) == 1) {
		absolute_strcpy(flist_folder, data_fname, rel_fname) ;
		if (parse_dataset(data_fname, det, frames)) {
			fclose(fp) ;
			return -1 ;
		}
	}
	else {
		fprintf(stderr, "No datasets found in %s\n", flist) ;
		fclose(fp) ;
		return -1 ;
	}
	
	curr = frames ;
	frames->tot_num_data = frames->num_data ;
	frames->tot_mean_count = frames->num_data * frames->mean_count ;
	frames->num_data_prev = 0 ;
	int num_datasets = 1 ;
	
	while (fscanf(fp, "%s\n", rel_fname) == 1) {
		if (strlen(rel_fname) == 0)
			continue ;
		absolute_strcpy(flist_folder, data_fname, rel_fname) ;
		curr->next = malloc(sizeof(struct dataset)) ;
		curr = curr->next ;
		curr->next = NULL ;
		
		//fprintf(stderr, "%s[%d]: %d\n", data_fname, num_datasets, det[0].mapping[num_datasets]) ;
		if (parse_dataset(data_fname, &(det[det[0].mapping[num_datasets]]), curr)) {
			fclose(fp) ;
			return -1 ;
		}
		
		curr->num_data_prev = frames->tot_num_data ;
		frames->tot_num_data += curr->num_data ;
		frames->tot_mean_count += curr->num_data * curr->mean_count ;
		num_datasets++ ;
	}
	fclose(fp) ;
	
	frames->tot_mean_count /= frames->tot_num_data ;
	calc_sum_fact(det, frames) ;
	
	return num_datasets ;
}

void generate_blacklist(char *config_fname, struct dataset *frames) {
	char blacklist_fname[1024] = {'\0'}, sel_string[1024] = {'\0'} ;
	char line[1024], section_name[1024], config_folder[1024], *token ;
	char *temp_fname = strndup(config_fname, 1024) ;
	sprintf(config_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "emc") == 0) {
			if (strcmp(token, "blacklist_file") == 0)
				absolute_strcpy(config_folder, blacklist_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "selection") == 0)
				strcpy(sel_string, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	if (sel_string[0] == '\0') {
		make_blacklist(blacklist_fname, 0, frames) ;
	}
	else if (strcmp(sel_string, "odd_only") == 0) {
		fprintf(stderr, "Only processing 'odd' frames\n") ;
		make_blacklist(blacklist_fname, 1, frames) ;
	}
	else if (strcmp(sel_string, "even_only") == 0) {
		fprintf(stderr, "Only processing 'even' frames\n") ;
		make_blacklist(blacklist_fname, 2, frames) ;
	}
	else {
		fprintf(stderr, "Did not understand selection keyword: %s. Will process all frames\n", sel_string) ;
		make_blacklist(blacklist_fname, 0, frames) ;
	}
	
	fprintf(stderr, "%d/%d blacklisted frames\n", frames->num_blacklist, frames->tot_num_data) ;
}

void make_blacklist(char *fname, int flag, struct dataset *frames) {
	int d, current = flag%2 ;
	frames->num_blacklist = 0 ;
	frames->blacklist = calloc(frames->tot_num_data, sizeof(uint8_t)) ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp != NULL) {
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			fscanf(fp, "%" SCNu8 "\n", &frames->blacklist[d]) ;
			if (frames->blacklist[d])
				frames->num_blacklist++ ;
		}
		fclose(fp) ;
	}
	
	if (flag > 0) {
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (!frames->blacklist[d]) {
			frames->blacklist[d] = current ;
			frames->num_blacklist += current ;
			current = 1 - current ;
		}
	}
}

int write_dataset(struct dataset *frames) {
	int header[253] = {0} ;
	FILE *fp = fopen(frames->filename, "wb") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to open file for writing, %s\n", frames->filename) ;
		return 1 ;
	}
	fwrite(&frames->num_data, sizeof(int), 1, fp) ;
	fwrite(&frames->num_pix, sizeof(int), 1, fp) ;
	fwrite(&frames->type, sizeof(int), 1, fp) ;
	fwrite(header, sizeof(int), 253, fp) ;
	
	if (frames->type == 0) {
		fwrite(frames->ones, sizeof(int), frames->num_data, fp) ;
		fwrite(frames->multi, sizeof(int), frames->num_data, fp) ;
		fwrite(frames->place_ones, sizeof(int), frames->ones_total, fp) ;
		fwrite(frames->place_multi, sizeof(int), frames->multi_total, fp) ;
		fwrite(frames->count_multi, sizeof(int), frames->multi_total, fp) ;
	}
	else if (frames->type == 1) {
		fwrite(frames->int_frames, sizeof(int), frames->num_data*frames->num_pix, fp) ;
	}
	else if (frames->type == 2) {
		fwrite(frames->frames, sizeof(double), frames->num_data*frames->num_pix, fp) ;
	}
	fclose(fp) ;
	
	return 0 ;
}

void free_data(int scale_flag, struct dataset *frames) {
	struct dataset *temp, *curr = frames ;
	
	if (frames == NULL)
		return ;
	
	if (scale_flag)
		free(frames->count) ;
	if (frames->blacklist != NULL)
		free(frames->blacklist) ;
	if (frames->sum_fact != NULL)
		free(frames->sum_fact) ;
	
	while (curr != NULL) {
		if (curr->type == 0) {
			free(curr->ones) ;
			free(curr->multi) ;
			free(curr->ones_accum) ;
			free(curr->multi_accum) ;
			free(curr->place_ones) ;
			free(curr->place_multi) ;
			free(curr->count_multi) ;
		}
		else if (curr->type == 1) {
			free(curr->int_frames) ;
		}
		else if (curr->type == 2) {
			free(curr->frames) ;
		}
		temp = curr ;
		curr = curr->next ;
		free(temp) ;
	}
}

