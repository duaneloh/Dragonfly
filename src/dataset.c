#include "dataset.h"

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

int parse_data(char *fname, struct detector *det, struct dataset *frames) {
	struct dataset *curr ;
	char data_fname[1024] ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "data_flist %s not found. Exiting.\n", fname) ;
		return -1 ;
	}
	
	if (fscanf(fp, "%s\n", data_fname) == 1) {
		if (parse_dataset(data_fname, det, frames))
			return -1 ;
	}
	else {
		fprintf(stderr, "No datasets found in %s\n", fname) ;
		return -1 ;
	}
	
	curr = frames ;
	frames->tot_num_data = frames->num_data ;
	frames->tot_mean_count = frames->num_data * frames->mean_count ;
	frames->num_data_prev = 0 ;
	int num_datasets = 1 ;
	
	while (fscanf(fp, "%s\n", data_fname) == 1) {
		if (strlen(data_fname) == 0)
			continue ;
		curr->next = malloc(sizeof(struct dataset)) ;
		curr = curr->next ;
		curr->next = NULL ;
		
		//fprintf(stderr, "%s[%d]: %d\n", data_fname, num_datasets, det[0].mapping[num_datasets]) ;
		if (parse_dataset(data_fname, &(det[det[0].mapping[num_datasets]]), curr))
			return -1 ;
		
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

void free_data(int scale_flag, struct dataset *frames) {
	struct dataset *curr = frames ;
	while (curr != NULL) {
		if (curr->type == 0) {
			free(curr->ones) ;
			free(curr->place_ones) ;
			free(curr->multi) ;
			free(curr->place_multi) ;
			free(curr->count_multi) ;
		}
		else if (curr->type == 1) {
			free(curr->int_frames) ;
		}
		else if (curr->type == 2) {
			free(curr->frames) ;
		}
		curr = curr->next ;
	}
	
	if (scale_flag)
		free(frames->count) ;
	if (frames->blacklist != NULL)
		free(frames->blacklist) ;
	if (frames->sum_fact != NULL)
		free(frames->sum_fact) ;
}

