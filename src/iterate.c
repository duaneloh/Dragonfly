#include "iterate.h"

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

int generate_iterate(char *config_fname, char *config_section, int continue_flag, double qmax, struct params *param, struct detector *det, struct dataset *dset, struct iterate *iter) {
	FILE *fp ;
	double model_mean ;
	char input_fname[1024] = {'\0'}, scale_fname[1024] = {'\0'} ;
	char line[1024], section_name[1024], config_folder[1024], *token ;
	char *temp_fname = strndup(config_fname, 1024) ;
	sprintf(config_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	iter->size = -1 ;
	iter->scale = NULL ;
	iter->modes = param->modes ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, config_section) == 0) {
			if (strcmp(token, "size") == 0)
				iter->size = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "start_model_file") == 0)
				absolute_strcpy(config_folder, input_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "scale_file") == 0)
				absolute_strcpy(config_folder, scale_fname, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	calculate_size(qmax, iter) ;
	
	if (continue_flag) {
		fp = fopen(param->log_fname, "r") ;
		if (fp == NULL) {
			fprintf(stderr, "No log file found to continue run\n") ;
			return 1 ;
		}
		else {
			while (!feof(fp))
				fgets(line, 1024, fp) ;
			sscanf(line, "%d", &param->start_iter) ;
			fclose(fp) ;
			
			sprintf(input_fname, "%s/output/intens_%.3d.bin", param->output_folder, param->start_iter) ;
			if (param->need_scaling)
				sprintf(scale_fname, "%s/scale/scale_%.3d.dat", param->output_folder, param->start_iter) ;
			param->start_iter += 1 ;
			fprintf(stderr, "Continuing from previous run starting from iteration %d.\n", param->start_iter) ;
		}
	}
	
	if (param->need_scaling) {
		if (!param->rank && param->start_iter == 1) {
			sprintf(line, "%s/scale/scale_000.dat", param->output_folder) ;
			calc_scale(dset, det, line, iter) ;
		}
		else {
			calc_scale(dset, det, NULL, iter) ;
		}
		param->known_scale = parse_scale(scale_fname, iter) ;
	}
	
	if (!param->rank && param->start_iter == 1) {
		sprintf(line, "%s/output/intens_000.bin", param->output_folder) ;
		model_mean = dset[0].mean_count / det[0].rel_num_pix * 2. ;
#ifdef FIXED_SEED
		model_mean *= -1. ;
#endif // FIXED_SEED
		parse_input(input_fname, model_mean, line, param->rank, param->recon_type, iter) ;
	}
	else {
		parse_input(input_fname, 1., NULL, param->rank, param->recon_type, iter) ;
	}
	
	return 0 ;
}

void calculate_size(double qmax, struct iterate *iter) {
	if (iter->size < 0) {
		iter->size = 2*ceil(qmax) + 3 ;
		fprintf(stderr, "Calculated 3D volume size = %ld\n", iter->size) ;
	}
	else {
		fprintf(stderr, "Provided 3D volume size = %ld\n", iter->size) ;
	}
	iter->center = iter->size / 2 ;
}

int parse_scale(char *fname, struct iterate *iter) {
	int flag = 0 ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Using uniform scale factors\n") ;
	}
	else {
		fprintf(stderr, "Using scale factors from %s\n", fname) ;
		flag = 1 ;
		int d ;
		for (d = 0 ; d < iter->tot_num_data ; ++d)
			fscanf(fp, "%lf", &iter->scale[d]) ;
		fclose(fp) ;
	}
	
	return flag ;
}

void calc_scale(struct dataset *frames, struct detector *det, char* print_fname, struct iterate *iter) {
	int d, t ;
	struct dataset *curr ;
	curr = frames ;
	
	iter->tot_num_data = frames->tot_num_data ;
	iter->scale = calloc(iter->tot_num_data, sizeof(double)) ;
	frames->count = calloc(iter->tot_num_data, sizeof(int)) ;
	
	while (curr != NULL) {
		if (curr->type == 0) {
			for (d = 0 ; d < curr->num_data ; ++d) {
				iter->scale[curr->num_data_prev + d] = 1. ;
				for (t = 0 ; t < curr->ones[d] ; ++t)
				if (det->mask[curr->place_ones[curr->ones_accum[d] + t]] < 1)
					frames->count[curr->num_data_prev + d]++ ;
				
				for (t = 0 ; t < curr->multi[d] ; ++t)
				if (det->mask[curr->place_multi[curr->multi_accum[d] + t]] < 1)
					frames->count[curr->num_data_prev + d] += curr->count_multi[curr->multi_accum[d] + t] ;
			}
		}
		else if (curr->type == 1) {
			for (d = 0 ; d < curr->num_data ; ++d) {
				iter->scale[curr->num_data_prev + d] = 1. ;
				for (t = 0 ; t < curr->num_pix ; ++t)
					frames->count[curr->num_data_prev+d] += curr->int_frames[d*curr->num_pix + t] ;
			}
		}
		else if (curr->type == 2) {
			for (d = 0 ; d < curr->num_data ; ++d) {
				iter->scale[curr->num_data_prev + d] = 1. ;
				for (t = 0 ; t < curr->num_pix ; ++t)
					frames->count[curr->num_data_prev+d] += curr->frames[d*curr->num_pix + t] ;
			}
		}
		
		curr = curr->next ;
	}
	
	if (print_fname != NULL) {
		FILE *fp = fopen(print_fname, "w") ;
		for (d = 0 ; d < iter->tot_num_data ; ++d)
			fprintf(fp, "%.6e\n", iter->scale[d]) ;
		fclose(fp) ;
		fprintf(stderr, "Written initial scale factors to %s\n", print_fname) ;
	}
}

void normalize_scale(struct dataset *dset, struct iterate *iter) {
	double mean_scale = 0. ;
	long d, x, tot_vol = iter->modes * iter->vol ;
	
	for (d = 0 ; d < iter->tot_num_data ; ++d)
	if (!dset->blacklist[d])
		mean_scale += iter->scale[d] ;
	
	mean_scale /= iter->tot_num_data - dset->num_blacklist ;
	
	for (x = 0 ; x < tot_vol ; ++x)
		iter->model1[x] *= mean_scale ;
	
	for (d = 0 ; d < iter->tot_num_data ; ++d)
	if (!dset->blacklist[d])
		iter->scale[d] /= mean_scale ;
	
	iter->rms_change *= mean_scale ;
}

void parse_input(char *fname, double mean, char *print_fname, int rank, int recon_type, struct iterate *iter) {
	iter->vol = 0 ;
	if (recon_type == RECON3D)
		iter->vol = iter->size * iter->size * iter->size ;
	else if (recon_type == RECON2D)
		iter->vol = iter->size * iter->size ;
	long tot_vol = iter->modes * iter->vol ;
	
	iter->model1 = malloc(tot_vol * sizeof(double)) ;
	iter->model2 = malloc(tot_vol * sizeof(double)) ;
	iter->inter_weight = malloc(tot_vol * sizeof(double)) ;
	
	if (rank == 0) {
		FILE *fp = fopen(fname, "r") ;
		if (fp == NULL) {
			if (mean < 0.)
				fprintf(stderr, "Random start with fixed seed\n") ;
			else
				fprintf(stderr, "Random start\n") ;
			
			long x ;
			const gsl_rng_type *T ;
			gsl_rng_env_setup() ;
			T = gsl_rng_default ;
			gsl_rng *rng = gsl_rng_alloc(T) ;
			if (mean < 0.) {
				gsl_rng_set(rng, 0x5EED) ;
				mean *= -1. ;
			}
			else {
				struct timeval t ;
				gettimeofday(&t, NULL) ;
				gsl_rng_set(rng, t.tv_sec + t.tv_usec) ;
			}
			
			for (x = 0 ; x < tot_vol ; ++x)
				iter->model1[x] = gsl_rng_uniform(rng) * mean ;
			
			gsl_rng_free(rng) ;
		}
		else {
			fprintf(stderr, "Starting from %s\n", fname) ;
			
			fread(iter->model1, sizeof(double), tot_vol, fp) ;
			fclose(fp) ;
		}
		
		if (print_fname != NULL) {
			FILE *fp = fopen(print_fname, "wb") ;
			fwrite(iter->model1, sizeof(double), tot_vol, fp) ;
			fclose(fp) ;
		}
	}
}

void free_iterate(struct iterate *iter) {
	if (iter == NULL)
		return ;
	
	if (iter->model1 != NULL)
		free(iter->model1) ;
	if (iter->model2 != NULL)
		free(iter->model2) ;
	if (iter->inter_weight != NULL)
		free(iter->inter_weight) ;
	if (iter->scale != NULL)
		free(iter->scale) ;
	free(iter) ;
}
