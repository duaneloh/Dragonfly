#include "detector.h"

double parse_detector(char *fname, struct detector *det, int norm_flag) {
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

void free_detector(struct detector *det) {
	free(det->pixels) ;
	free(det->mask) ;
}
