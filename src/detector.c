#include "detector.h"

double parse_detector_core(char *fname, struct detector *det, int norm_flag) {
	int t, d ;
	double q, qmax = -1., mean_pol = 0. ;
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
	det->pixels = malloc(4 * det->num_pix * sizeof(double)) ;
	det->mask = malloc(det->num_pix * sizeof(uint8_t)) ;
	for (t = 0 ; t < det->num_pix ; ++t) {
		for (d = 0 ; d < 4 ; ++d)
			fscanf(fp, "%lf", &det->pixels[t*4 + d]) ;
		fscanf(fp, "%" SCNu8, &det->mask[t]) ;
		if (det->mask[t] < 1)
			det->rel_num_pix++ ;
		q = pow(det->pixels[t*4+0], 2.) + pow(det->pixels[t*4+1], 2.) + pow(det->pixels[t*4+2], 2.) ;
		if (q > qmax)
			qmax = q ;
		mean_pol += det->pixels[t*4 + 3] ;
	}
	
	if (norm_flag == 1) {
		mean_pol /= det->num_pix ;
		for (t = 0 ; t < det->num_pix ; ++t)
			det->pixels[t*4 + 3] /= mean_pol ;
	}
	
	fclose(fp) ;
	
	return sqrt(qmax) ;
}

double parse_detector(char *fname, struct detector *det, int norm_flag) {
    double qmax = parse_detector_core(fname, det, norm_flag);
    det->num_det = 1;
    det->num_dfiles = 1;
    return qmax;
}

double parse_detector_list(char *fname, struct detector **det_ptr, int norm_flag) {
	int j, num_det = 0, num_dfiles, new_det ;
	double det_qmax, qmax = -1. ;
	char name_list[1024][1024] ;
	int det_mapping[1024] = {0} ;
	struct detector *det ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to open in_detector_list %s\n", fname) ;
		return -1. ;
	}
	for (num_dfiles = 0 ; num_dfiles < 1024 ; ++num_dfiles) {
		if (feof(fp) || fscanf(fp, "%s\n", name_list[num_det]) != 1)
			break ;
		new_det = 1 ;
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
		det_qmax = parse_detector_core(name_list[j], &det[j], 1) ;
		if (det_qmax < 0.)
			return -1. ;
		if (det_qmax > qmax)
			qmax = det_qmax ;
	}
	
	return qmax ;
}

void free_detector(struct detector *det) {
	int detn ;
	for (detn = 0 ; detn < det[0].num_det ; ++detn) {
		free(det[detn].pixels) ;
		free(det[detn].mask) ;
	}
	free(det) ;
}
