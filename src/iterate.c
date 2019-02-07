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

static void calc_mean_counts(struct dataset *frames, struct detector *det, struct iterate *iter) {
	int dset = 0, detn ;
	struct dataset *curr = frames ;
	int *numd = calloc(det[0].num_det, sizeof(int)) ;
	
	while (curr != NULL) {
		detn = det[0].mapping[dset] ;
		iter->mean_count[detn] += curr->mean_count * curr->num_data ;
		numd[detn] += curr->num_data ;
		dset++ ;
		curr = curr->next ;
	}
	
	for (detn = 0 ; detn < det[0].num_det ; ++detn)
		iter->mean_count[detn] /= numd[detn] ;
	
	free(numd) ;
}

// Public functions below

int generate_iterate(char *config_fname, char *config_section, int continue_flag, double qmax, struct params *param, struct detector *det, struct dataset *dset, struct iterate *iter) {
	FILE *fp ;
	double model_mean ;
	char input_fname[2048] = {'\0'}, scale_fname[2048] = {'\0'} ;
	char probs_fname[2048] = {'\0'} ;
	char line[2048], section_name[1024], config_folder[1024], *token ;
	char *temp_fname = strndup(config_fname, 1024) ;
	sprintf(config_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	iter->size = -1 ;
	iter->scale = NULL ;
	iter->rel_quat = NULL ;
	iter->modes = param->modes ;
	iter->rescale = calloc(det[0].num_det, sizeof(double)) ;
	iter->mean_count = calloc(det[0].num_det, sizeof(double)) ;
	iter->tot_num_data = dset->tot_num_data ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 2048, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, config_section) == 0) {
			if (strcmp(token, "size") == 0)
				iter->size = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "start_model_file") == 0)
				absolute_strcpy(config_folder, input_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "scale_file") == 0)
				absolute_strcpy(config_folder, scale_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "probs_file") == 0)
				absolute_strcpy(config_folder, probs_fname, strtok(NULL, " =\n")) ;
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
				fgets(line, 2048, fp) ;
			sscanf(line, "%d", &param->start_iter) ;
			fclose(fp) ;
			
#ifdef WITH_HDF5
			sprintf(input_fname, "%s/output_%.3d.h5", param->output_folder, param->start_iter) ;
			if (param->need_scaling)
				sprintf(scale_fname, "%s/output_%.3d.h5", param->output_folder, param->start_iter) ;
			if (param->refine && probs_fname[0] == '\0')
				sprintf(probs_fname, "%s/output_%.3d.h5", param->output_folder, param->start_iter) ;
#else // WITH_HDF5
			sprintf(input_fname, "%s/output/intens_%.3d.bin", param->output_folder, param->start_iter) ;
			if (param->need_scaling)
				sprintf(scale_fname, "%s/scale/scale_%.3d.dat", param->output_folder, param->start_iter) ;
			if (param->refine && probs_fname[0] == '\0')
				sprintf(probs_fname, "%s/probabilities/probabilities_%.3d.emc", param->output_folder, param->start_iter) ;
#endif // WITH_HDF5
			param->start_iter += 1 ;
			fprintf(stderr, "Continuing from previous run starting from iteration %d.\n", param->start_iter) ;
		}
	}
	
	if (param->need_scaling) {
		calc_scale(dset, det, iter) ;
		param->known_scale = parse_scale(scale_fname, iter) ;
		
		if (param->update_scale == 0 && param->known_scale == 0) {
			fprintf(stderr, "If not updating scales, need input scale_file.\n") ;
			return 1 ;
		}
		else if (param->update_scale == 0) {
			fprintf(stderr, "Not updating these scale factors\n") ;
		}
	}
	
	if (param->refine) {
		if (probs_fname[0] == '\0') {
			fprintf(stderr, "Need coarse probabilities in probs_file to do refinement\n") ;
			return 1 ;
		}
		struct rotation *qcoarse = calloc(1, sizeof(struct rotation)) ;
		struct rotation *qfine = calloc(1, sizeof(struct rotation)) ;
		if (quat_gen(param->coarse_div, qcoarse) < 0) {
			fprintf(stderr, "Problem generating quat[%d]\n", param->coarse_div) ;
			return 1 ;
		}
		if (quat_gen(param->fine_div, qfine) < 0) {
			fprintf(stderr, "Problem generating quat[%d]\n", param->fine_div) ;
			return 1 ;
		}
		
		iter->quat_mapping = malloc(qfine->num_rot * sizeof(int)) ;
		voronoi_subset(qcoarse, qfine, iter->quat_mapping) ;
		
		if (parse_rel_quat(probs_fname, qcoarse->num_rot, 0, iter))
			return 1 ;
		
		free_quat(qcoarse) ;
		free_quat(qfine) ;
	}
	
	model_mean = dset[0].mean_count / det[0].rel_num_pix * 2. ;
#ifdef FIXED_SEED
	model_mean *= -1. ;
#endif // FIXED_SEED
	parse_input(input_fname, model_mean, param->rank, param->recon_type, iter) ;
	calc_powder(dset, det, iter) ;
	calc_mean_counts(dset, det, iter) ;
	
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
	char line[8], hdfheader[8] = {137, 'H', 'D', 'F', '\r', '\n', 26, '\n'} ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Using uniform scale factors\n") ;
	}
	else {
		fprintf(stderr, "Using scale factors from %s\n", fname) ;
		flag = 1 ;
		fread(line, sizeof(char), 8, fp) ;
		if (strncmp(line, hdfheader, 8) == 0) {
#ifdef WITH_HDF5
			fclose(fp) ;
			hid_t file, dset, dspace ;
			file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT) ;
			dset = H5Dopen(file, "/scale", H5P_DEFAULT) ;
			dspace = H5Dget_space(dset) ;
			if (H5Sget_simple_extent_npoints(dspace) != iter->tot_num_data) {
				fprintf(stderr, "Number of frames in 'scale' dataset does not match. ") ;
				fprintf(stderr, "Defaulting to uniform scale factors\n") ;
				return 0 ;
			}
			H5Dread(dset, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, iter->scale) ;
			H5Dclose(dset) ;
			H5Fclose(file) ;
#else // WITH_HDF5
			fprintf(stderr, "H5 output support not compiled. Cannot parse scale factors\n") ;
			fprintf(stderr, "Defaulting to uniform scale factors\n") ;
			return 0 ;
#endif // WITH_HDF5
		}
		else {
			fseek(fp, 0, SEEK_SET) ;
			int d ;
			for (d = 0 ; d < iter->tot_num_data ; ++d)
				fscanf(fp, "%lf", &iter->scale[d]) ;
			fclose(fp) ;
		}
	}
	
	return flag ;
}

void calc_powder(struct dataset *frames, struct detector *det, struct iterate *iter) {
	int dset = 0, detn, d, t, pixel ;
	int *nframes = calloc(det[0].num_det, sizeof(int)) ;
	struct dataset *curr = frames ;
	
	for (detn = 0 ; detn < det[0].num_det ; ++detn)
		det[detn].powder = calloc(det[detn].num_pix, sizeof(double)) ;
	
	while (curr != NULL) {
		detn = det[0].mapping[dset] ;
		for (d = 0 ; d < curr->num_data ; ++d) {
			nframes[detn]++ ;
			for (t = 0 ; t < curr->ones[d] ; ++t) {
				pixel = curr->place_ones[curr->ones_accum[d] + t] ;
				if (det[detn].mask[pixel] < 1)
					det[detn].powder[pixel]++ ;
			}
			for (t = 0 ; t < curr->multi[d] ; ++t) {
				pixel = curr->place_multi[curr->multi_accum[d] + t] ;
				if (det[detn].mask[pixel] < 1)
					det[detn].powder[pixel] += curr->count_multi[curr->multi_accum[d] + t] ;
			}
		}
		
		dset++ ;
		curr = curr->next ;
	}
	
	for (detn = 0 ; detn < det[0].num_det ; ++detn)
		for (t = 0 ; t < det[detn].num_pix ; ++t)
			det[detn].powder[t] /= nframes[detn] ;
	free(nframes) ;
}

void calc_scale(struct dataset *frames, struct detector *det, struct iterate *iter) {
	int dset = 0, detn, d, t ;
	struct dataset *curr ;
	curr = frames ;
	
	iter->scale = calloc(iter->tot_num_data, sizeof(double)) ;
	frames->count = calloc(iter->tot_num_data, sizeof(int)) ;
	
	while (curr != NULL) {
		detn = det[0].mapping[dset] ;
		if (curr->type == 0) {
			for (d = 0 ; d < curr->num_data ; ++d) {
				iter->scale[curr->num_data_prev + d] = 1. ;
				for (t = 0 ; t < curr->ones[d] ; ++t)
				if (det[detn].mask[curr->place_ones[curr->ones_accum[d] + t]] < 1)
					frames->count[curr->num_data_prev + d]++ ;
				
				for (t = 0 ; t < curr->multi[d] ; ++t)
				if (det[detn].mask[curr->place_multi[curr->multi_accum[d] + t]] < 1)
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
		
		dset++ ;
		curr = curr->next ;
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

void parse_input(char *fname, double mean, int rank, int recon_type, struct iterate *iter) {
	iter->vol = 0 ;
	if (recon_type == RECON3D)
		iter->vol = iter->size * iter->size * iter->size ;
	else if (recon_type == RECON2D || recon_type == RECONRZ)
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
			
#ifdef WITH_HDF5
			fclose(fp) ;
			hid_t file, dset ;
			file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT) ;
			dset = H5Dopen(file, "/intens", H5P_DEFAULT) ;
			H5Dread(dset, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, iter->model1) ;
			H5Dclose(dset) ;
			H5Fclose(file) ;
#else // WITH_HDF5
			fread(iter->model1, sizeof(double), tot_vol, fp) ;
			fclose(fp) ;
#endif // WITH_HDF5
		}
	}
}

int parse_rel_quat(char *fname, int num_rot_coarse, int parse_prob, struct iterate *iter) {
	int num_rot_prob ;
	char line[8], hdfheader[8] = {137, 'H', 'D', 'F', '\r', '\n', 26, '\n'} ;
	
	FILE *fp = fopen(fname, "rb") ;
	if (fp == NULL) {
		fprintf(stderr, "Could not open rel_quat file %s\n", fname) ;
		return 1 ;
	}
	fread(line, sizeof(char), 8, fp) ;
	fclose(fp) ;
	
	if (strncmp(line, hdfheader, 8) == 0) {
#ifdef WITH_HDF5
		fprintf(stderr, "Parsing rel_quat from H5 output file\n") ;
		hid_t file, dset, dspace, dtype ;
		hsize_t d, num_data ;
		hvl_t *buffer ;
		
		file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT) ;
		if (file < 0) {
			fprintf(stderr, "Could not open rel_quat file %s\n", fname) ;
			return 1 ;
		}
		
		dset = H5Dopen(file, "probabilities/num_rot", H5P_DEFAULT) ;
		H5Dread(dset, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_rot_prob) ;
		H5Dclose(dset) ;
		if (num_rot_prob != num_rot_coarse) {
			fprintf(stderr, "Rotation sampling in probs file does not match num_rot_coarse\n") ;
			fprintf(stderr, "%d vs %d\n", num_rot_prob, num_rot_coarse) ;
			return 1 ;
		}
		
		dset = H5Dopen(file, "probabilities/place", H5P_DEFAULT) ;
		dtype = H5Tvlen_create(H5T_STD_I32LE) ;
		dspace = H5Dget_space(dset) ;
		H5Sget_simple_extent_dims(dspace, &num_data, NULL) ;
		if ((int) num_data != iter->tot_num_data) {
			fprintf(stderr, "num_data of rel_quat (%lld) does not match %d\n", num_data, iter->tot_num_data) ;
			return 1 ;
		}
		
		buffer = malloc(num_data * sizeof(hvl_t)) ;
		iter->num_rel_quat = malloc(num_data * sizeof(int)) ;
		iter->rel_quat = malloc(num_data * sizeof(int*)) ;
		
		H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) ;
		for (d = 0 ; d < num_data ; ++d) {
			iter->num_rel_quat[d] = buffer[d].len ;
			iter->rel_quat[d] = buffer[d].p ;
		}
		
		free(buffer) ;
		H5Dclose(dset) ;
		H5Tclose(dtype) ;
		H5Sclose(dspace) ;
		
		if (parse_prob) {
			fprintf(stderr, "Parsing sparse probabilities\n") ;
			dset = H5Dopen(file, "probabilities/prob", H5P_DEFAULT) ;
			dtype = H5Tvlen_create(H5T_IEEE_F64LE) ;
			dspace = H5Dget_space(dset) ;
			H5Sget_simple_extent_dims(dspace, &num_data, NULL) ;
			
			buffer = malloc(num_data * sizeof(hvl_t)) ;
			iter->rel_prob = malloc(num_data * sizeof(double*)) ;
			
			H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) ;
			for (d = 0 ; d < num_data ; ++d)
				iter->rel_prob[d] = buffer[d].p ;
			
			free(buffer) ;
			H5Dclose(dset) ;
			H5Tclose(dtype) ;
			H5Sclose(dspace) ;
		}
		
		H5Fclose(file) ;
#else // WITH_HDF5
		fprintf(stderr, "H5 output file support not compiled. Cannot parse rel_quat.\n") ;
		return 1 ;
#endif // WITH_HDF5
	}
	else {
		fprintf(stderr, "Parsing rel_quat from emc file\n") ;
		int d, num_data ;
		
		fp = fopen(fname, "rb") ;
		fread(&num_data, sizeof(int), 1, fp) ;
		if (num_data != iter->tot_num_data) {
			fprintf(stderr, "num_data of rel_quat (%d) does not match\n", num_data) ;
			fclose(fp) ;
			return 1 ;
		}
		fread(&num_rot_prob, sizeof(int), 1, fp) ;
		if (num_rot_prob != num_rot_coarse) {
			fprintf(stderr, "Rotation sampling in probs file does not match num_rot_coarse\n") ;
			fprintf(stderr, "%d vs %d\n", num_rot_prob, num_rot_coarse) ;
			fclose(fp) ;
			return 1 ;
		}
		iter->num_rel_quat = malloc(num_data * sizeof(int)) ;
		iter->rel_quat = malloc(num_data * sizeof(int*)) ;
		
		fseek(fp, 1024, SEEK_SET) ;
		fread(iter->num_rel_quat, sizeof(int), num_data, fp) ;
		for (d = 0 ; d < num_data ; ++d) {
			iter->rel_quat[d] = malloc(iter->num_rel_quat[d] * sizeof(int)) ;
			fread(iter->rel_quat[d], sizeof(int), iter->num_rel_quat[d], fp) ;
		}
		if (parse_prob) {
			fprintf(stderr, "Parsing sparse probabilities\n") ;
			iter->rel_prob = malloc(num_data * sizeof(double*)) ;
			for (d = 0 ; d < num_data ; ++d) {
				iter->rel_prob[d] = malloc(iter->num_rel_quat[d] * sizeof(int)) ;
				fread(iter->rel_prob[d], sizeof(int), iter->num_rel_quat[d], fp) ;
			}
		}
		
		fclose(fp) ;
	}
	
	return 0 ;
}

void free_iterate(struct iterate *iter) {
	int d ;
	
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
	if (iter->rescale != NULL)
		free(iter->rescale) ;
	if (iter->rel_quat != NULL) {
		for (d = 0 ; d < iter->tot_num_data ; ++d)
			free(iter->rel_quat[d]) ;
		free(iter->rel_quat) ;
		free(iter->num_rel_quat) ;
	}
	free(iter) ;
}
