#include "emc.h"

int parse_det(char *fname) {
	int t ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "pixel_fname %s not found. Exiting...1\n", fname) ;
		return 1 ;
	}
	fscanf(fp, "%d %lf", &num_pix, &detd) ;
	det = malloc(4 * num_pix * sizeof(double)) ;
	for (t = 0 ; t < 4 * num_pix ; ++t) 
		fscanf(fp, "%lf", &det[t]) ;
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
	for (r = 0 ; r < num_rot ; ++r)
	for (t = 0 ; t < 5 ; ++t) {
		fscanf(fp, "%lf", &quat[r*5 + t]) ;
		if (t == 4)
			total_weight += quat[r*5 + 4] ;
	}
	total_weight = 1. / total_weight ;
	for (r = 0 ; r < num_rot ; ++r)
		quat[r*5 + 4] *= total_weight ;
	fclose(fp) ;
	
	if (num_proc > 1) {
		num_rot_p = (int) num_rot / num_proc ;
		int num_proc_rem = num_rot % num_proc ;
		if (rank <= num_proc_rem) {
			num_rot_p++ ;
			num_rot_shift = rank * num_rot_p ;
		}
		else
			num_rot_shift = rank * num_rot_p + num_proc_rem ;
	}
	else {
		num_rot_p = num_rot ;
		num_rot_shift = 0 ;
	}
	
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
	fread(&(current->mean_count), sizeof(double) , 1, fp) ;
	
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
	
	// Correct mean count in the presence of mask
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
	fprintf(stderr, "tot_photons = %.6e\n", tot_mean_count) ;
	
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
		fprintf(stderr, "tot_photons = %.6e\n", tot_mean_count) ;
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
			fprintf(stderr, "Random start\n") ;
			model_mean = tot_mean_count / rel_num_pix ;
			for (x = 0 ; x < size * size * size ; ++x)
				model1[x] = ((double) rand() / RAND_MAX) * model_mean ;
		}
		else {
			fprintf(stderr, "Using %s\n", fname) ;
			fread(model1, sizeof(double), size * size * size, fp) ;
			fclose(fp) ;
		}
	}
}

int parse_mask(char *fname) {
	int t ;
	
	FILE *fp = fopen(fname, "rb") ;
	if (fp == NULL) {
		fprintf(stderr, "mask_fname, %s not found. Exiting.\n", fname) ;
		return 1 ;
	}
	mask = malloc(num_pix * sizeof(uint8_t)) ;
	fread(mask, sizeof(uint8_t), num_pix, fp) ;
	fclose(fp) ;
	
	rel_num_pix = 0 ;
	for (t = 0 ; t < num_pix ; ++t)
	if (mask[t] < 1)
		rel_num_pix++ ;
	
	return 0 ;
}

void calc_scale() {
	long d_counter = 0, ones_counter, multi_counter ;
	int d, t ;
	struct dataset *curr ;
	curr = frames ;
	double inner_mean_count = 0. ;
	
	scale = calloc(tot_num_data, sizeof(double)) ;
	count = calloc(tot_num_data, sizeof(int)) ;
	
	while (curr != NULL) {
		ones_counter = 0 ;
		multi_counter = 0 ;
		for (d = d_counter ; d < d_counter + curr->num_data ; ++d) {
			for (t = 0 ; t < curr->ones[d-d_counter] ; ++t) {
				if (mask[curr->place_ones[ones_counter + t]] == 1)
					scale[d]++ ;
				else if (mask[curr->place_ones[ones_counter + t]] < 1)
					count[d]++ ;
			}
			
			for (t = 0 ; t < curr->multi[d-d_counter] ; ++t) {
				if (mask[curr->place_multi[multi_counter + t]] == 1)
					scale[d] += curr->count_multi[multi_counter + t] ;
				else if (mask[curr->place_multi[multi_counter + t]] < 1)
					count[d] += curr->count_multi[multi_counter + t] ;
			}
			
			inner_mean_count += scale[d] ;
			ones_counter += curr->ones[d - d_counter] ;
			multi_counter += curr->multi[d - d_counter] ;
		}
		
		d_counter += curr->num_data ;
		curr = curr->next ;
	}
	
	inner_mean_count /= tot_num_data ;
	for (d = 0 ; d < tot_num_data ; ++d)
		scale[d] /= inner_mean_count ;
	
	FILE *fp = fopen("phi/phi000.dat", "w") ;
	for (d = 0 ; d < tot_num_data ; ++d)
		fprintf(fp, "%.6e\n", scale[d]) ;
	fclose(fp) ;
}

int factorial(int num) {
	if (num < 2)
		return 1 ;
	else
		return num * factorial(num-1) ;
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
			if (mask[curr->place_multi[multi_counter]] < 1)
				sum_fact[d] += log(factorial(curr->count_multi[multi_counter++])) ;
		}
		
		d_counter += curr->num_data ;
		curr = curr->next ;
	}
}

void gen_blacklist(char *fname) {
	int d ;
	num_blacklist = 0 ;
	blacklist = calloc(tot_num_data, sizeof(uint8_t)) ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		if (rank == 0)
			fprintf(stderr, "No blacklist file found. All frames whitelisted\n") ;
/*			fprintf(stderr, "No blacklist file found. Using even orig. frames\n") ;
		for (d = 0 ; d < tot_num_data ; ++d)
		if ((d/16) % 2 == 0) {
			blacklist[d] = 1 ;
			num_blacklist++ ;
		}
*/	}
	else {
		for (d = 0 ; d < tot_num_data ; ++d) {
			fscanf(fp, "%" SCNu8 "\n", &blacklist[d]) ;
			if (blacklist[d])
				num_blacklist++ ;
		}
	}
	fprintf(stderr, "%d blacklisted frames\n", num_blacklist) ;
}

void parse_scale(char *fname) {
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		if (rank == 0)
			fprintf(stderr, "Using uniform scale factors\n") ;
	}
	else {
		fprintf(stderr, "Using scale factors from %s\n", fname) ;
		known_scale = 1 ;
		int d ;
		for (d = 0 ; d < tot_num_data ; ++d)
			fscanf(fp, "%lf", &scale[d]) ;
		fclose(fp) ;
	}
}

int setup() {
	FILE *fp ;
	char pixel_fname[999], quat_fname[999], mask_fname[999] ;
	char data_flist[999], input_fname[999] ;
	char scale_fname[999], blacklist_fname[999] ;
	
	known_scale = 0 ;
	
	char line[999], *token ;
	fp = fopen("config.ini", "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file config.ini not found.\n") ;
		return 1 ;
	}
	while (fgets(line, 999, fp) != NULL) {
		token = strtok(line, " =") ;
		if (token[0] == '#' || token[0] == '\n' || token[0] == '[')
			continue ;
		
		if (strcmp(token, "size") == 0)
			size = atoi(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "center") == 0)
			center = atoi(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "need_scaling") == 0)
			need_scaling = atoi(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "alpha") == 0)
			alpha = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "beta") == 0)
			beta = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "data") == 0)
			strcpy(data_flist, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "output_prefix") == 0)
			strcpy(output_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "input") == 0)
			strcpy(input_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "detector") == 0)
			strcpy(pixel_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "quaternion") == 0)
			strcpy(quat_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "mask") == 0)
			strcpy(mask_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "blacklist") == 0)
			strcpy(blacklist_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "scale") == 0)
			strcpy(scale_fname, strtok(NULL, " =\n")) ;
	}
	fclose(fp) ;
	fprintf(stderr, "Parsed config.ini\n") ;
	
	if (parse_det(pixel_fname))
		return 1 ;
	if (parse_quat(quat_fname))
		return 1 ;
	if (parse_mask(mask_fname))
		return 1 ;
	if (need_scaling) {
		calc_scale() ;
		parse_scale(scale_fname) ;
	}
	if (parse_data(data_flist))
		return 1 ;
	
	calc_sum_fact() ;
	parse_input(input_fname) ;
	gen_blacklist(blacklist_fname) ;
	
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
