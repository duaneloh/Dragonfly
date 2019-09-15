#include "detector.h"

static double preprocess_detector(struct detector *det, int norm_flag) {
	int t, stride ;
	double q, qmax = -1., mean_pol = 0. ;
	
	det->rel_num_pix = 0 ;
	if (norm_flag < 0)
		stride = 3 ;
	else
		stride = 4 ;
	
	for (t = 0 ; t < det->num_pix ; ++t) {
		if (det->mask[t] < 1)
			det->rel_num_pix++ ;
		if (det->mask[t] < 2) {
			if (stride == 3)
				q = pow(det->pixels[t*3+0], 2.) + pow(det->pixels[t*3+1], 2.) ;
			else
				q = pow(det->pixels[t*stride+0], 2.) + pow(det->pixels[t*stride+1], 2.) + pow(det->pixels[t*stride+2], 2.) ;
			if (q > qmax)
				qmax = q ;
		}
		mean_pol += det->pixels[t*stride + stride-1] ;
	}
	
	if (norm_flag == 1 || norm_flag < 0) {
		mean_pol /= det->num_pix ;
		for (t = 0 ; t < det->num_pix ; ++t)
			det->pixels[t*stride + stride-1] /= mean_pol ;
	}
	
	det->background = calloc(det->num_pix, sizeof(double)) ;
	
	return qmax ;
}

static double parse_asciidetector(char *fname, struct detector *det, int norm_flag) {
	int t, d ;
	double temp, qmax ;
	char line[1024] ;
	
	FILE *fp = fopen(fname, "r") ;
	fgets(line, 1024, fp) ;
	sscanf(line, "%d %lf %lf\n", &det->num_pix, &det->detd, &det->ewald_rad) ;
	if (norm_flag >= 0) {
		det->pixels = malloc(4 * det->num_pix * sizeof(double)) ;
		det->mask = malloc(det->num_pix * sizeof(uint8_t)) ;
		for (t = 0 ; t < det->num_pix ; ++t) {
			for (d = 0 ; d < 4 ; ++d)
				fscanf(fp, "%lf", &det->pixels[t*4 + d]) ;
			fscanf(fp, "%" SCNu8, &det->mask[t]) ;
		}
		
		qmax = preprocess_detector(det, norm_flag) ;
	}
	else {
		if (det->detd == 0. || det->ewald_rad == 0.) {
			fprintf(stderr, "Need new format detector to create 2D detector\n") ;
			fclose(fp) ;
			return -1. ;
		}
		fprintf(stderr, "Creating 2D detector\n") ;
		det->pixels = malloc(3 * det->num_pix * sizeof(double)) ;
		det->mask = malloc(det->num_pix * sizeof(uint8_t)) ;
		for (t = 0 ; t < det->num_pix ; ++t) {
			fscanf(fp, "%lf", &det->pixels[t*3 + 0]) ;
			fscanf(fp, "%lf", &det->pixels[t*3 + 1]) ;
			fscanf(fp, "%lf", &temp) ;
			fscanf(fp, "%lf", &det->pixels[t*3 + 2]) ;
			fscanf(fp, "%" SCNu8, &det->mask[t]) ;
			
			// Mapping 3D q-space voxels to 2D
			det->pixels[t*3+0] *= det->detd / (temp + det->ewald_rad) ;
			det->pixels[t*3+1] *= det->detd / (temp + det->ewald_rad) ;
		}
		
		qmax = preprocess_detector(det, norm_flag) ;
	}
	fclose(fp) ;
	
	return sqrt(qmax) ;
}

#ifdef WITH_HDF5
static double parse_h5detector(char *fname, struct detector *det, int norm_flag) {
	hid_t file, dset, dtype, dspace, mspace ;
	int i, ndims ;
	double qmax, temp ;
	det->num_pix = 1 ;
	
	file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT) ;
	dset = H5Dopen(file, "/qx", H5P_DEFAULT) ;
	dtype = H5Dget_type(dset) ;
	dspace = H5Dget_space(dset) ;
	ndims = H5Sget_simple_extent_ndims(dspace) ;
	
	hsize_t dims[ndims] ;
	H5Sget_simple_extent_dims(dspace, dims, NULL) ;
	for (i = 0 ; i < ndims ; ++i)
		det->num_pix *= dims[i] ;
	fprintf(stderr, "H5 detector with %d pixels\n", det->num_pix) ;
	
	det->pixels = calloc(4 * det->num_pix, sizeof(double)) ;
	det->mask = malloc(det->num_pix * sizeof(uint8_t)) ;
	hsize_t pixdims[1] = {4*det->num_pix}, start[1] = {0}, count[1] = {det->num_pix}, stride[1] = {4} ;
	mspace = H5Screate_simple(1, pixdims, NULL) ;
	H5Sselect_hyperslab(mspace, H5S_SELECT_SET, start, stride, count, NULL) ;
	H5Dread(dset, dtype, mspace, dspace, H5P_DEFAULT, det->pixels) ;
	
	dset = H5Dopen(file, "/qy", H5P_DEFAULT) ;
	start[0] = 1 ;
	H5Sselect_hyperslab(mspace, H5S_SELECT_SET, start, stride, count, NULL) ;
	H5Dread(dset, dtype, mspace, dspace, H5P_DEFAULT, det->pixels) ;
	H5Dclose(dset) ;
	
	dset = H5Dopen(file, "/qz", H5P_DEFAULT) ;
	start[0] = 2 ;
	H5Sselect_hyperslab(mspace, H5S_SELECT_SET, start, stride, count, NULL) ;
	H5Dread(dset, dtype, mspace, dspace, H5P_DEFAULT, det->pixels) ;
	H5Dclose(dset) ;
	
	dset = H5Dopen(file, "/corr", H5P_DEFAULT) ;
	start[0] = 3 ;
	H5Sselect_hyperslab(mspace, H5S_SELECT_SET, start, stride, count, NULL) ;
	H5Dread(dset, dtype, mspace, dspace, H5P_DEFAULT, det->pixels) ;
	H5Dclose(dset) ;
	
	dset = H5Dopen(file, "/mask", H5P_DEFAULT) ;
	dtype = H5Dget_type(dset) ;
	size_t typesize = H5Tget_size(dtype) ;
	dspace = H5Dget_space(dset) ;
	if (typesize == 1) {
		uint8_t *mask_buffer = det->mask ;
		H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, mask_buffer) ;
	}
	else if (typesize == 2) {
		uint16_t *mask_buffer = malloc(det->num_pix * typesize) ;
		H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, mask_buffer) ;
		for (i = 0 ; i < det->num_pix ; ++i)
			det->mask[i] = (uint8_t) mask_buffer[i] ;
	}
	else if (typesize == 4) {
		uint32_t *mask_buffer = malloc(det->num_pix * typesize) ;
		H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, mask_buffer) ;
		for (i = 0 ; i < det->num_pix ; ++i)
			det->mask[i] = (uint8_t) mask_buffer[i] ;
	}
	else {
		uint64_t *mask_buffer = malloc(det->num_pix * typesize) ;
		H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, mask_buffer) ;
		for (i = 0 ; i < det->num_pix ; ++i)
			det->mask[i] = (uint8_t) mask_buffer[i] ;
	}
	H5Dclose(dset) ;
	
	dset = H5Dopen(file, "/detd", H5P_DEFAULT) ;
	dtype = H5Dget_type(dset) ;
	H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(det->detd)) ;
	
	dset = H5Dopen(file, "/ewald_rad", H5P_DEFAULT) ;
	dtype = H5Dget_type(dset) ;
	H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(det->ewald_rad)) ;
	
	if (norm_flag < 0) { // 2D detector
		for (i = 0 ; i < det->num_pix ; ++i) {
			det->pixels[i*3+0] = det->pixels[i*4+0] ;
			det->pixels[i*3+1] = det->pixels[i*4+1] ;
			temp = det->pixels[i*4+2] ;
			det->pixels[i*3+2] = det->pixels[i*4+3] ;
			
			det->pixels[i*3+0] *= det->detd / (temp + det->ewald_rad) ;
			det->pixels[i*3+1] *= det->detd / (temp + det->ewald_rad) ;
		}
	}
	
	qmax = preprocess_detector(det, norm_flag) ;
	
	return sqrt(qmax) ;
}
#endif //WITH_HDF5

static int parse_background(char *fname, struct detector *det) {
	char line[8], hdfheader[8] = {137, 'H', 'D', 'F', '\r', '\n', 26, '\n'} ;
	FILE *fp ;
	
	fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Could not find background file %s\n", fname) ;
		return 1 ;
	}
	fread(line, 8, sizeof(char), fp) ;
	fclose(fp) ;
	
	if (strncmp(line, hdfheader, 8) == 0) {
#ifdef WITH_HDF5
		fprintf(stderr, "Parsing HDF5 background file\n") ;
		hid_t file, dset ;
		file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT) ;
		dset = H5Dopen(file, "/background", H5P_DEFAULT) ;
		H5Dread(dset, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, det->background) ;
		H5Dclose(dset) ;
		H5Fclose(file) ;
#else
		fprintf(stderr, "H5 background file support not compiled\n") ;
		return 1 ;
#endif // WITH_HDF5
	}
	else {
		fp = fopen(fname, "r") ;
		fread(det->background, sizeof(double), det->num_pix, fp) ;
		fclose(fp) ;
	}

	return 0 ;
}

static int parse_background_list(char *flist, struct detector **det_list) {
	int i = 0, ndet = (*det_list)[0].num_det ;
	char abs_fname[1024] ;
	char flist_folder[1024], rel_fname[1024] ;
	char *temp_fname = strndup(flist, 1024) ;
	sprintf(flist_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	
	FILE *fp = fopen(flist, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to open background_list %s\n", flist) ;
		return 1 ;
	}
	while (fscanf(fp, "%1023s\n", rel_fname) == 1) {
		absolute_strcpy(flist_folder, abs_fname, rel_fname) ;
		if (i < ndet) {
			if (parse_background(abs_fname, det_list[i])) {
				fclose(fp) ;
				return 1 ;
			}
		}
		i++ ;
	}
	if (i != ndet) {
		fprintf(stderr, "Mismatch of number of background and unique detector files (%d vs %d)\n", i, ndet) ;
		fclose(fp) ;
		return 1 ;
	}
	fclose(fp) ;
	
	return 0 ;
}

double detector_from_config(char *config_fname, char *config_section, struct detector **det_list, int norm_flag) {
	double qmax ;
	char det_fname[1024] = {'\0'}, det_flist[1024] = {'\0'}, out_det_fname[1024] = {'\0'} ;
	char bg_fname[1024] = {'\0'}, bg_flist[1024] = {'\0'} ;
	char line[1024], section_name[1024], config_folder[1024], *token ;
	char *temp_fname = strndup(config_fname, 1024) ;
	sprintf(config_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "make_detector") == 0) {
			if (strcmp(token, "out_detector_file") == 0)
				absolute_strcpy(config_folder, out_det_fname, strtok(NULL, " =\n")) ;
		}
		else if (strcmp(section_name, config_section) == 0) {
			if (strcmp(token, "in_detector_file") == 0)
				absolute_strcpy(config_folder, det_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "in_detector_list") == 0)
				absolute_strcpy(config_folder, det_flist, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "background_file") == 0)
				absolute_strcpy(config_folder, bg_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "background_list") == 0)
				absolute_strcpy(config_folder, bg_flist, strtok(NULL, " =\n")) ;
		}
	}
	fclose(config_fp) ;
	
	if (strcmp(det_fname, "make_detector:::out_detector_file") == 0)
		strcpy(det_fname, out_det_fname) ;
	
	if (det_flist[0] != '\0' && det_fname[0] != '\0') {
		fprintf(stderr, "Both in_detector_file and in_detector_list specified. Pick one.\n") ;
		return -1. ;
	}
	else if (det_fname[0] != '\0') {
		fprintf(stderr, "Parsing detector file: %s\n", det_fname) ;
		*det_list = calloc(1, sizeof(struct detector)) ;
		(*det_list)[0].num_det = 1 ;
		(*det_list)[0].num_dfiles = 0 ;
		memset((*det_list)[0].mapping, 0, 1024*sizeof(int)) ;
		if ((qmax = parse_detector(det_fname, det_list[0], norm_flag)) < 0.)
			return qmax ;
	}
	else if (det_flist[0] != '\0') {
		if ((qmax = parse_detector_list(det_flist, det_list, norm_flag)) < 0.)
			return qmax ;
	}
	else {
		fprintf(stderr, "Need either in_detector_file or in_detector_list.\n") ;
		return -1. ;
	}
	
	if (bg_flist[0] != '\0' && bg_fname[0] != '\0') {
		fprintf(stderr, "Both background_file and background_list specified. Pick one.\n") ;
		return -1. ;
	}
	else if (bg_fname[0] != '\0') {
		if ((*det_list)[0].num_det > 1)
			fprintf(stderr, "Multiple detectors and single background file. Assuming same background for all detectors\n") ;
		fprintf(stderr, "Parsing background file: %s\n", bg_fname) ;
		(*det_list)[0].with_bg = 1 ;
		if (parse_background(bg_fname, det_list[0]))
			return -1. ;
	}
	else if (bg_flist[0] != '\0') {
		fprintf(stderr, "Parsing background file list: %s\n", bg_flist) ;
		(*det_list)[0].with_bg = 1 ;
		if (parse_background_list(bg_flist, det_list))
			return -1. ;
	}
	
	fprintf(stderr, "Number of unique detectors = %d\n", (*det_list)[0].num_det) ;
	fprintf(stderr, "Number of detector files in list = %d\n", (*det_list)[0].num_dfiles) ;
	
	return qmax ;
}

double parse_detector(char *fname, struct detector *det, int norm_flag) {
	double qmax = -1. ;
	char line[1024], hdfheader[8] = {137, 'H', 'D', 'F', '\r', '\n', 26, '\n'} ;
	
	det->rel_num_pix = 0 ;
	det->detd = 0. ;
	det->ewald_rad = 0. ;
	det->powder = NULL ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "det_fname %s not found. Exiting...1\n", fname) ;
		return -1. ;
	}
	fread(line, 8, sizeof(char), fp) ;
	fclose(fp) ;
	
	if (strncmp(line, hdfheader, 8) == 0) {
#ifdef WITH_HDF5
		qmax = parse_h5detector(fname, det, norm_flag) ;
#else
		fprintf(stderr, "H5 detector support not compiled\n") ;
		return -1. ;
#endif // WITH_HDF5
	}
	else {
		qmax = parse_asciidetector(fname, det, norm_flag) ;
	}

	return qmax ;
}

double parse_detector_list(char *flist, struct detector **det_ptr, int norm_flag) {
	int j, num_det = 0, num_dfiles, new_det, norm_all = 0 ;
	double det_qmax, qmax = -1. ;
	char name_list[1024][1024] ;
	char flist_folder[1024], rel_fname[1024] ;
	int det_mapping[1024] = {0} ;
	struct detector *det ;
	
	char *temp_fname = strndup(flist, 1024) ;
	sprintf(flist_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	
	FILE *fp = fopen(flist, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to open in_detector_list %s\n", flist) ;
		return -1. ;
	}
	for (num_dfiles = 0 ; num_dfiles < 1024 ; ++num_dfiles) {
		if (feof(fp) || fscanf(fp, "%1023s\n", rel_fname) != 1)
			break ;
		new_det = 1 ;
		absolute_strcpy(flist_folder, name_list[num_det], rel_fname) ;
		for (j = 0 ; j < num_det ; ++j)
		if (strcmp(name_list[num_det], name_list[j]) == 0) {
			new_det = 0 ;
			det_mapping[num_dfiles] = j ;
			break ;
		}
		if (new_det) {
			det_mapping[num_dfiles] = num_det ;
			num_det++ ;
		}
	}
	
	// If multiple detectors require normalization, do it for all detectors together
	if (norm_flag == 1 && num_det > 1) {
		norm_flag = 0 ;
		norm_all = 1 ;
	}
	
	*det_ptr = calloc(num_det, sizeof(struct detector)) ;
	det = *det_ptr ;
	memcpy(det[0].mapping, det_mapping, 1024*sizeof(int)) ;
	det[0].num_det = num_det ;
	det[0].num_dfiles = num_dfiles ;
	for (j = 0 ; j < num_det ; ++j) {
		det_qmax = parse_detector(name_list[j], &det[j], norm_flag) ;
		if (det_qmax < 0.) {
			fclose(fp) ;
			return -1. ;
		}
		if (det_qmax > qmax)
			qmax = det_qmax ;
	}
	fclose(fp) ;
	
	if (norm_all) {
		// norm_all only activated for norm_flag == 1 which implies 3D detector
		fprintf(stderr, "Normalizing corr over all detectors\n") ;
		double mean_pol = 0. ;
		int t, tot_num_pix = 0 ;
		for (j = 0 ; j < num_det ; ++j) {
			tot_num_pix += det[j].num_pix ;
			for (t = 0 ; t < det[j].num_pix ; ++t)
				mean_pol += det[j].pixels[t*4+3] ;
		}
		mean_pol /= tot_num_pix ;
		for (j = 0 ; j < num_det ; ++j)
		for (t = 0 ; t < det[j].num_pix ; ++t)
			det[j].pixels[t*4+3] /= mean_pol ;
	}
	
	return qmax ;
}

void copy_detector(struct detector *in_det, struct detector *out_det) {
	out_det->num_pix = in_det->num_pix ;
	out_det->rel_num_pix = in_det->rel_num_pix ;
	out_det->detd = in_det->detd ;
	out_det->ewald_rad = in_det->ewald_rad ;
	out_det->num_det = in_det->num_det ;
	out_det->num_dfiles = in_det->num_dfiles ;
	out_det->with_bg = in_det->with_bg ;
	
	out_det->pixels = malloc(out_det->num_pix * 4 * sizeof(double)) ;
	out_det->mask = malloc(out_det->num_pix * sizeof(uint8_t)) ;
	memcpy(out_det->pixels, in_det->pixels, out_det->num_pix*4*sizeof(double)) ;
	memcpy(out_det->mask, in_det->mask, out_det->num_pix*sizeof(uint8_t)) ;
	
	if (in_det->powder != NULL) {
		out_det->powder = malloc(out_det->num_pix * sizeof(double)) ;
		memcpy(out_det->powder, in_det->powder, out_det->num_pix*sizeof(double)) ;
	}
	
	if (in_det->with_bg) {
		out_det->background = malloc(out_det->num_pix * sizeof(double)) ;
		memcpy(out_det->background, in_det->background, out_det->num_pix*sizeof(double)) ;
	}
}

void remask_detector(struct detector *det, double radius) {
	int t ;
	double q, *pix ;
	
	for (t = 0 ; t < det->num_pix ; ++t) {
		pix = &(det->pixels[t*4]) ;
		q = sqrt(pix[0]*pix[0] + pix[1]*pix[1] * pix[2]*pix[2]) ;
		if (det->mask[t] == 0 && q > radius) {
			det->mask[t] = 1 ;
			det->rel_num_pix -= 1 ;
		}
	}
}

void free_detector(struct detector *det) {
	int detn ;
	
	if (det == NULL)
		return ;
	
	for (detn = 0 ; detn < det[0].num_det ; ++detn) {
		free(det[detn].pixels) ;
		free(det[detn].mask) ;
		if (det[detn].powder != NULL)
			free(det[detn].powder) ;
	}
	free(det) ;
}
