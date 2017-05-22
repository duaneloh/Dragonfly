#include "dataset.h"

void calc_sum_fact(struct detector *det, struct dataset *frames) {
	int d, t, d_counter = 0 ;
	long multi_counter ;
	struct dataset *curr = frames ;
	
	frames->sum_fact = calloc(frames->tot_num_data, sizeof(double)) ;
	
	while (curr != NULL) {
		multi_counter = 0 ;
		
		for (d = d_counter ; d < d_counter + curr->num_data ; ++d) {
			for (t = 0 ; t < curr->multi[d - d_counter] ; ++t)
			if (det->mask[curr->place_multi[multi_counter + t]] < 1)
				frames->sum_fact[d] += gsl_sf_lnfact(curr->count_multi[multi_counter + t]) ;
			multi_counter += curr->multi[d - d_counter] ;
		}
		
		d_counter += curr->num_data ;
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
		if (det->mask[current->place_ones[ones_counter + t]] < 1)
			current->mean_count += 1. ;
		
		for (t = 0 ; t < current->multi[d] ; ++t)
		if (det->mask[current->place_multi[multi_counter + t]] < 1)
			current->mean_count += current->count_multi[multi_counter + t] ;
		
		ones_counter += current->ones[d] ;
		multi_counter += current->multi[d] ;
	}
	
	current->mean_count /= current->num_data ;
	current->tot_mean_count = current->mean_count ;
	
	return 0 ;
}

int parse_data(char *fname, struct detector *det, struct dataset *frames) {
	struct dataset *curr ;
	char data_fname[999] ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "data_flist %s not found. Exiting.\n", fname) ;
		return 1 ;
	}
	
	if (fscanf(fp, "%s\n", data_fname) == 1) {
		if (parse_dataset(data_fname, det, frames))
			return 1 ;
	}
	else {
		fprintf(stderr, "No datasets found in %s\n", fname) ;
		return 1 ;
	}
	
	curr = frames ;
	frames->tot_num_data = frames->num_data ;
	frames->tot_mean_count = frames->num_data * frames->mean_count ;
	int num_datasets = 1 ;
	
	while (fscanf(fp, "%s\n", data_fname) == 1) {
		if (strlen(data_fname) == 0)
			continue ;
		curr->next = malloc(sizeof(struct dataset)) ;
		curr = curr->next ;
		curr->next = NULL ;
		
		if (parse_dataset(data_fname, det, curr))
			return 1 ;
		
		frames->tot_num_data += curr->num_data ;
		frames->tot_mean_count += curr->num_data * curr->mean_count ;
		num_datasets++ ;
	}
	fclose(fp) ;
	
	frames->tot_mean_count /= frames->tot_num_data ;
	calc_sum_fact(det, frames) ;
	
	return 0 ;
}

void gen_blacklist(char *fname, int flag, struct dataset *frames) {
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

void free_data(int scale_flag, struct dataset *frames) {
	struct dataset *curr = frames ;
	while (curr != NULL) {
		free(curr->ones) ;
		free(curr->place_ones) ;
		free(curr->multi) ;
		free(curr->place_multi) ;
		free(curr->count_multi) ;
		curr = curr->next ;
	}
	
	if (scale_flag)
		free(frames->count) ;
	free(frames->blacklist) ;
	free(frames->sum_fact) ;
}

