#include "detector.h"

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

double generate_detectors(char *config_fname, char *config_section, struct detector **det_list, int norm_flag) {
	double qmax ;
	char det_fname[1024] = {'\0'}, det_flist[1024] = {'\0'}, out_det_fname[1024] = {'\0'} ;
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
		*det_list = malloc(sizeof(struct detector)) ;
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
	
	fprintf(stderr, "Number of unique detectors = %d\n", (*det_list)[0].num_det) ;
	fprintf(stderr, "Number of detector files in list = %d\n", (*det_list)[0].num_dfiles) ;
	
	return qmax ;
}

double parse_detector(char *fname, struct detector *det, int norm_flag) {
	int t, d ;
	double temp, q, qmax = -1., mean_pol = 0. ;
	char line[1024] ;
	
	det->rel_num_pix = 0 ;
	det->detd = 0. ;
	det->ewald_rad = 0. ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "det_fname %s not found. Exiting...1\n", fname) ;
		return -1. ;
	}
	fgets(line, 1024, fp) ;
	sscanf(line, "%d %lf %lf\n", &det->num_pix, &det->detd, &det->ewald_rad) ;
	if (norm_flag >= 0) {
		det->pixels = malloc(4 * det->num_pix * sizeof(double)) ;
		det->mask = malloc(det->num_pix * sizeof(uint8_t)) ;
		for (t = 0 ; t < det->num_pix ; ++t) {
			for (d = 0 ; d < 4 ; ++d)
				fscanf(fp, "%lf", &det->pixels[t*4 + d]) ;
			fscanf(fp, "%" SCNu8, &det->mask[t]) ;
			
			if (det->mask[t] < 1)
				det->rel_num_pix++ ;
			if (det->mask[t] < 2) {
				q = pow(det->pixels[t*4+0], 2.) + pow(det->pixels[t*4+1], 2.) + pow(det->pixels[t*4+2], 2.) ;
				if (q > qmax)
					qmax = q ;
			}
			mean_pol += det->pixels[t*4 + 3] ;
		}
		
		if (norm_flag == 1) {
			mean_pol /= det->num_pix ;
			for (t = 0 ; t < det->num_pix ; ++t)
				det->pixels[t*4 + 3] /= mean_pol ;
		}
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
			
			if (det->mask[t] < 1)
				det->rel_num_pix++ ;
			if (det->mask[t] < 2) {
				q = pow(det->pixels[t*3+0], 2.) + pow(det->pixels[t*3+1], 2.) ;
				if (q > qmax)
					qmax = q ;
			}
			mean_pol += det->pixels[t*3 + 2] ;
		}
		
		mean_pol /= det->num_pix ;
		for (t = 0 ; t < det->num_pix ; ++t)
			det->pixels[t*3 + 2] /= mean_pol ;
	}
	fclose(fp) ;
	
	return sqrt(qmax) ;
}

double parse_detector_list(char *flist, struct detector **det_ptr, int norm_flag) {
	int j, num_det = 0, num_dfiles, new_det ;
	double det_qmax, qmax = -1. ;
	char name_list[1024][1024] ;
	char flist_folder[1024] ;
	int det_mapping[1024] = {0} ;
	struct detector *det ;
	
	char *temp_fname = strndup(flist, 1024) ;
	sprintf(flist_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	char rel_fname[1024] ;
	
	FILE *fp = fopen(flist, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to open in_detector_list %s\n", flist) ;
		return -1. ;
	}
	for (num_dfiles = 0 ; num_dfiles < 1024 ; ++num_dfiles) {
		if (feof(fp) || fscanf(fp, "%s\n", rel_fname) != 1)
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
		//fprintf(stderr, "mapping[%d] = %d/%d, %s\n", num_dfiles, det_mapping[num_dfiles], num_det, name_list[det_mapping[num_dfiles]]) ;
	}
	
	*det_ptr = malloc(num_det * sizeof(struct detector)) ;
	det = *det_ptr ;
	memcpy(det[0].mapping, det_mapping, 1024*sizeof(int)) ;
	det[0].num_det = num_det ;
	det[0].num_dfiles = num_dfiles ;
	//fprintf(stderr, "mapping: %d, %d, ...\n", det[0].mapping[0], det[0].mapping[1]) ;
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
	
	return qmax ;
}

void free_detector(struct detector *det) {
	int detn ;
	
	if (det == NULL)
		return ;
	
	for (detn = 0 ; detn < det[0].num_det ; ++detn) {
		free(det[detn].pixels) ;
		free(det[detn].mask) ;
	}
	free(det) ;
}
