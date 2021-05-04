#include "dataset.h"

static int parse_binarydataset(char *fname, struct detector *det, struct dataset *current) {
	int d ;
	
	FILE *fp = fopen(fname, "rb") ;
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

	return 0 ;
}

#ifdef WITH_HDF5
static int parse_h5dataset(char *fname, struct detector *det, struct dataset *current) {
	int d ;
	hid_t file, dset, dspace, dtype ;
	hsize_t bufsize ;
	hvl_t *buffer ;
	
	file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT) ;
	
	dset = H5Dopen(file, "num_pix", H5P_DEFAULT) ;
	H5Dread(dset, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(current->num_pix)) ;
	H5Dclose(dset) ;
	if (current->num_pix != det->num_pix)
		fprintf(stderr, "WARNING! The detector file and photons file %s do not "
		                "have the same number of pixels\n", current->filename) ;
	
	dset = H5Dopen(file, "place_ones", H5P_DEFAULT) ;
	dtype = H5Tvlen_create(H5T_STD_I32LE) ;
	dspace = H5Dget_space(dset) ;
	H5Sget_simple_extent_dims(dspace, &bufsize, NULL) ;
	current->num_data = bufsize ;
	current->tot_num_data = current->num_data ;
	buffer = malloc(current->num_data * sizeof(hvl_t)) ;
	H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) ;
	current->ones = malloc(current->num_data * sizeof(int)) ;
	current->ones_accum = malloc(current->num_data * sizeof(long)) ;
	H5Dvlen_get_buf_size(dset, dtype, dspace, &bufsize) ;
	current->ones_total = bufsize / sizeof(int) ;
	current->place_ones = malloc(current->ones_total * sizeof(int)) ;
	current->ones_accum[0] = 0 ;
	current->ones[0] = buffer[0].len ;
	memcpy(current->place_ones, buffer[0].p, sizeof(int)*buffer[0].len) ;
	free(buffer[0].p) ;
	for (d = 1 ; d < current->num_data ; ++d) {
		current->ones[d] = buffer[d].len ;
		current->ones_accum[d] = current->ones_accum[d-1] + current->ones[d-1] ;
		memcpy(&current->place_ones[current->ones_accum[d]], buffer[d].p, sizeof(int)*buffer[d].len) ;
		free(buffer[d].p) ;
	}
	if (current->ones_total != current->ones_accum[current->num_data-1] + current->ones[current->num_data-1])
		fprintf(stderr, "WARNING: ones_total mismatch in %s\n", current->filename) ;
	H5Dclose(dset) ;
	
	dset = H5Dopen(file, "place_multi", H5P_DEFAULT) ;
	H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) ;
	current->multi = malloc(current->num_data * sizeof(int)) ;
	current->multi_accum = malloc(current->num_data * sizeof(long)) ;
	H5Dvlen_get_buf_size(dset, dtype, dspace, &bufsize) ;
	current->multi_total = bufsize / sizeof(int) ;
	current->place_multi = malloc(current->multi_total * sizeof(int)) ;
	current->multi_accum[0] = 0 ;
	current->multi[0] = buffer[0].len ;
	memcpy(current->place_multi, buffer[0].p, sizeof(int)*buffer[0].len) ;
	free(buffer[0].p) ;
	for (d = 1 ; d < current->num_data ; ++d) {
		current->multi[d] = buffer[d].len ;
		current->multi_accum[d] = current->multi_accum[d-1] + current->multi[d-1] ;
		memcpy(&current->place_multi[current->multi_accum[d]], buffer[d].p, sizeof(int)*buffer[d].len) ;
		free(buffer[d].p) ;
	}
	if (current->multi_total != current->multi_accum[current->num_data-1] + current->multi[current->num_data-1])
		fprintf(stderr, "WARNING: multi_total mismatch in %s\n", current->filename) ;
	H5Dclose(dset) ;
	
	dset = H5Dopen(file, "count_multi", H5P_DEFAULT) ;
	H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) ;
	current->count_multi = malloc(current->multi_total * sizeof(int)) ;
	for (d = 0 ; d < current->num_data ; ++d) {
		memcpy(&current->count_multi[current->multi_accum[d]], buffer[d].p, sizeof(int)*buffer[d].len) ;
		free(buffer[d].p) ;
	}
	H5Dclose(dset) ;
	free(buffer) ;
	
	H5Sclose(dspace) ;
	H5Tclose(dtype) ;
	H5Fclose(file) ;
	
	return 0 ;
}
#endif // WITH_HDF5

static int get_val(void* buffer, int pixel, int valtype) {
	if (valtype == 0)
		return (int) ((uint8_t*) buffer)[pixel] ;
	else if (valtype == 1)
		return (int) ((int8_t*) buffer)[pixel] ;
	else if (valtype == 2)
		return (int) ((uint16_t*) buffer)[pixel] ;
	else if (valtype == 3)
		return (int) ((int16_t*) buffer)[pixel] ;
	else if (valtype == 4)
		return (int) ((uint32_t*) buffer)[pixel] ;
	else if (valtype == 5)
		return (int) ((int32_t*) buffer)[pixel] ;
	else if (valtype == 6)
		return (int) ((uint64_t*) buffer)[pixel] ;
	else
		return (int) ((int64_t*) buffer)[pixel] ;
}

#ifdef WITH_HDF5
static int parse_dense_dataset(char *fname, char *dset_name, struct detector *det, struct dataset *current) {
	int d, t, ndims, val, valtype = -1 ;
	hid_t file, dset, dspace, mspace, dtype ;
	hsize_t *bufsize, *single, *start, *stride ;
	H5T_sign_t sign ;
	size_t typesize ;
	void *buffer ;
	
	file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT) ;
	
	dset = H5Dopen(file, dset_name, H5P_DEFAULT) ;
	dspace = H5Dget_space(dset) ;
	// Get ndims
	ndims = H5Sget_simple_extent_ndims(dspace) ;
	if (ndims < 2) {
		fprintf(stderr, "Need frames dataset in %s to be at least 2-dimensional\n", fname) ;
		return 1 ;
	}
	fprintf(stderr, "%d dimensional dataset\n", ndims) ;
	
	// Get dims
	bufsize = malloc(ndims * sizeof(hsize_t)) ;
	H5Sget_simple_extent_dims(dspace, bufsize, NULL) ;
	current->num_data = bufsize[0] ;
	current->tot_num_data = current->num_data ;
	current->num_pix = 1 ;
	for (d = 1 ; d < ndims ; ++d)
		current->num_pix *= bufsize[d] ;
	if (current->num_pix != det->num_pix)
		fprintf(stderr, "WARNING! The detector file and photons file %s do not "
		                "have the same number of pixels\n", current->filename) ;
	fprintf(stderr, "%d pixels/frame\n", current->num_pix) ;
	
	// Allocate dataset memory
	current->ones = malloc(current->num_data * sizeof(int)) ;
	current->ones_accum = calloc(current->num_data, sizeof(long)) ;
	current->ones_total = current->num_data * current->num_pix / 10 ;
	current->place_ones = malloc(current->ones_total * sizeof(int)) ;
	
	current->multi = malloc(current->num_data * sizeof(int)) ;
	current->multi_accum = calloc(current->num_data, sizeof(long)) ;
	current->multi_total = current->num_data * current->num_pix / 100 ;
	current->place_multi = malloc(current->multi_total * sizeof(int)) ;
	current->count_multi = malloc(current->multi_total * sizeof(int)) ;
	fprintf(stderr, "Allocated memory\n") ;
	
	// Get datatype and allocate single frame
	dtype = H5Dget_type(dset) ;
	if (H5Tget_class(dtype) != H5T_INTEGER) {
		fprintf(stderr, "Photons need to be stored as an integer data type\n") ;
		return 1 ;
	}
	typesize = H5Tget_size(dtype) ;
	buffer = malloc(current->num_pix * typesize) ;
	sign = H5Tget_sign(dtype) ;
	fprintf(stderr, "Type params: %ld %d\n", typesize, sign) ;
	if (typesize == 1) {
		if (sign == H5T_SGN_NONE)
			valtype = 0 ;
		else
			valtype = 1 ;
	}
	else if (typesize == 2) {
		if (sign == H5T_SGN_NONE)
			valtype = 2 ;
		else
			valtype = 3 ;
	}
	else if (typesize == 4) {
		if (sign == H5T_SGN_NONE)
			valtype = 4 ;
		else
			valtype = 5 ;
	}
	else if (typesize == 8) {
		if (sign == H5T_SGN_NONE)
			valtype = 6 ;
		else
			valtype = 7 ;
	}
	if (valtype == -1) {
		fprintf(stderr, "Unknown data type in %s : %s\n", fname, dset_name) ;
		return 1 ;
	}
	fprintf(stderr, "valtype = %d\n", valtype) ;
	fprintf(stderr, "Number of frames = %d\n", current->num_data) ;
	//return 1 ;
	
	// Parse and sparsify each frame
	single = malloc(ndims * sizeof(hsize_t)) ;
	stride = malloc(ndims * sizeof(hsize_t)) ;
	start = calloc(ndims, sizeof(hsize_t)) ;
	for (d = 0 ; d < ndims ; ++d) {
		single[d] = bufsize[d] ;
		stride[d] = 1 ;
	}
	single[0] = 1 ;
	mspace = H5Screate_simple(ndims, single, NULL) ;
	for (d = 0 ; d < current->num_data ; ++d) {
		if (current->ones_accum[d] + current->num_pix > current->ones_total)  {
			current->ones_total += current->num_pix ;
			current->place_ones = realloc(current->place_ones, current->ones_total * sizeof(int)) ;
		}
		if (current->multi_accum[d] + current->num_pix > current->multi_total)  {
			current->multi_total += current->num_pix ;
			current->place_multi = realloc(current->place_multi, current->multi_total * sizeof(int)) ;
			current->count_multi = realloc(current->count_multi, current->multi_total * sizeof(int)) ;
		}
		start[0] = d ;
		H5Sselect_hyperslab(dspace, H5S_SELECT_SET, start, stride, single, NULL) ;
		H5Dread(dset, dtype, mspace, dspace, H5P_DEFAULT, buffer) ;
		for (t = 0 ; t < current->num_pix ; ++t)
		if (det->mask[t] < 2) {
			val = get_val(buffer, t, valtype) ;
			if (val == 1) {
				current->place_ones[current->ones_accum[d] + current->ones[d]] = t ;
				current->ones[d]++ ;
			}
			else if (val > 1) {
				current->place_multi[current->multi_accum[d] + current->multi[d]] = t ;
				current->count_multi[current->multi_accum[d] + current->multi[d]] = val ;
				current->multi[d]++ ;
			}
		}
		current->ones_accum[d+1] = current->ones_accum[d] + current->ones[d] ;
		current->multi_accum[d+1] = current->multi_accum[d] + current->multi[d] ;
	}
	
	H5Sclose(dspace) ;
	H5Tclose(dtype) ;
	H5Fclose(file) ;
	
	return 0 ;
}
#endif // WITH_HDF5

static int num_words(char *line) {
	int retval = 0 ;
	char *token = strtok(line, " \t\n") ;
	while (token != NULL) {
		retval++ ;
		token = strtok(NULL, " \t\n") ;
	}
	return retval ;
}

int data_from_config(char *config_fname, char *config_section, char *type_string, struct detector *det_list, struct dataset *frames_list) {
	int num_datasets = 0 ;
	char data_fname[1024] = {'\0'}, data_flist[1024] = {'\0'}, out_data_fname[1024] = {'\0'} ;
	char fname_opt[64], flist_opt[64] ;
	char line[1024] = {'\0'}, section_name[1024], config_folder[1024], *token ;
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
		fprintf(stderr, "Config file contains both %s and %s. Pick one.\n", fname_opt, flist_opt) ;
		return 1 ;
	}
	else if (data_fname[0] != '\0') {
		if (frames_list == NULL)
			frames_list = calloc(1, sizeof(struct dataset)) ;
		if (parse_dataset(data_fname, det_list, frames_list))
			return 1 ;
		frames_list->num_data_prev = 0 ;
		frames_list->next = NULL ;
		calc_sum_fact(det_list, frames_list) ;
		num_datasets = 1 ;
	}
	else if (data_flist[0] != '\0') {
		if (frames_list == NULL)
			frames_list = calloc(1, sizeof(struct dataset)) ;
		frames_list->next = NULL ;
		if ((num_datasets = parse_dataset_list(data_flist, det_list, frames_list)) < 0)
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
	int dset = 0, d, t, detn ;
	struct dataset *curr = frames ;
	
	frames->sum_fact = calloc(frames->tot_num_data, sizeof(double)) ;
	
	while (curr != NULL) {
		detn = det[0].mapping[dset] ;
		
		if (curr->type == 0) {
			for (d = 0 ; d < curr->num_data ; ++d)
			for (t = 0 ; t < curr->multi[d] ; ++t)
			if (det[detn].mask[curr->place_multi[curr->multi_accum[d] + t]] < 1)
				frames->sum_fact[curr->num_data_prev+d] += gsl_sf_lnfact(curr->count_multi[curr->multi_accum[d] + t]) ;
		}
		else if (curr->type == 1) {
			for (d = 0 ; d < curr->num_data ; ++d)
			for (t = 0 ; t < curr->num_pix ; ++t)
			if (det[detn].mask[t] < 1)
				frames->sum_fact[curr->num_data_prev+d] += gsl_sf_lnfact(curr->int_frames[d*curr->num_pix + t]) ;
		}
		else if (curr->type == 2) {
			for (d = 0 ; d < curr->num_data ; ++d)
				frames->sum_fact[curr->num_data_prev+d] = 0. ;
		}
		
		dset++ ;
		curr = curr->next ;
	}
}

int parse_dataset(char *fname, struct detector *det, struct dataset *current) {
	int err ;
	long d, t ;
	char line[1024], hdfheader[8] = {137, 'H', 'D', 'F', '\r', '\n', 26, '\n'} ;
	
	current->ones_total = 0, current->multi_total = 0 ;
	strcpy(current->filename, fname) ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "data_fname %s not found. Exiting.\n", fname) ;
		return 1 ;
	}
	fread(line, 8, sizeof(char), fp) ;
	fclose(fp) ;
	
	if (strncmp(line, hdfheader, 8) == 0) {
#ifdef WITH_HDF5
		fprintf(stderr, "Parsing HDF5 dataset %s\n", fname) ;
		current->type = 0 ;
		err = parse_h5dataset(fname, det, current) ;
#else
		fprintf(stderr, "H5 dataset support not compiled\n") ;
		return 1 ;
#endif // WITH_HDF5
	}
	else {
		err = parse_binarydataset(fname, det, current) ;
	}
	if (err)
		return err ;
	
	// Calculate mean count in the presence of mask
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
	
	return err ;
}

int parse_dataset_list(char *flist, struct detector *det, struct dataset *frames) {
	int num_sparse = 0, num_dense = 0, retval ;
	struct dataset *curr ;
	char line[2048] ;
	char *dset_name, data_fname[1024] ;
	char flist_folder[1024], *rel_fname ;
	char *temp_fname = strndup(flist, 1024) ;
	sprintf(flist_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	
	FILE *fp = fopen(flist, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "data_flist %s not found. Exiting.\n", flist) ;
		return -1 ;
	}
	
	if (fgets(line, 2048, fp) != NULL) {
		retval = num_words(line) ;
		rel_fname = strtok(line, " \t\n") ;
		absolute_strcpy(flist_folder, data_fname, rel_fname) ;
		if (retval > 1) {
#ifndef WITH_HDF5
			fprintf(stderr, "Dense dataset (%s : %s) needs HDF5 support\n", data_fname, dset_name) ;
			return -1 ;
#endif // WITH_HDF5
			dset_name = strtok(NULL, " \t\n") ;
			if (parse_dense_dataset(data_fname, dset_name, det, frames)) {
				fclose(fp) ;
				return -1 ;
			}
			num_dense++ ;
		}
		else {
			if (parse_dataset(data_fname, det, frames)) {
				fclose(fp) ;
				return -1 ;
			}
			num_sparse++ ;
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
	
	while (fgets(line, 2048, fp) != NULL) {
		retval = num_words(line) ;
		rel_fname = strtok(line, " \t\n") ;
		if (strlen(rel_fname) == 0)
			continue ;
		absolute_strcpy(flist_folder, data_fname, rel_fname) ;
		curr->next = calloc(1, sizeof(struct dataset)) ;
		curr = curr->next ;
		curr->next = NULL ;
		
		if (retval > 1) {
#ifndef WITH_HDF5
			fprintf(stderr, "Dense dataset (%s : %s) needs HDF5 support\n", data_fname, dset_name) ;
			return -1 ;
#endif // WITH_HDF5
			dset_name = strtok(NULL, " \t\n") ;
			if (parse_dense_dataset(data_fname, dset_name, &(det[det[0].mapping[num_sparse+num_dense]]), curr)) {
				fclose(fp) ;
				return -1 ;
			}
			num_dense++ ;
		}
		else {
			if (parse_dataset(data_fname, &(det[det[0].mapping[num_sparse+num_dense]]), curr)) {
				fclose(fp) ;
				return -1 ;
			}
			num_sparse++ ;
		}
		
		curr->num_data_prev = frames->tot_num_data ;
		frames->tot_num_data += curr->num_data ;
		frames->tot_mean_count += curr->num_data * curr->mean_count ;
	}
	fclose(fp) ;
	
	frames->tot_mean_count /= frames->tot_num_data ;
	calc_sum_fact(det, frames) ;
	
	fprintf(stderr, "(%d, %d) sparse and dense datasets\n", num_sparse, num_dense) ;
	return num_sparse + num_dense ;
}

void blacklist_from_config(char *config_fname, char *config_section, struct dataset *frames) {
	char blacklist_fname[1024] = {'\0'}, sel_string[1024] = {'\0'} ;
	char line[1024], section_name[1024], config_folder[1024], *token ;
	char *temp_fname = strndup(config_fname, 1024) ;
	sprintf(config_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, config_section) == 0) {
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

void calc_powder(struct detector *det, struct dataset *frames) {
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

