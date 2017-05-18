#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "detector.h"

int parse_detector(char *fname, struct detector **det_pointer) {
	int t, d ;
	double mean_pol = 0. ;
	struct detector *det = *det_pointer ;
	
	det->rel_num_pix = 0 ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "det_fname %s not found. Exiting...1\n", fname) ;
		return 1 ;
	}
	fscanf(fp, "%d", &det->num_pix) ;
	det->pixels = malloc(4 * det->num_pix * sizeof(double)) ;
	det->mask = malloc(det->num_pix * sizeof(uint8_t)) ;
	for (t = 0 ; t < det->num_pix ; ++t) {
		for (d = 0 ; d < 4 ; ++d)
			fscanf(fp, "%lf", &det->pixels[t*4 + d]) ;
		fscanf(fp, "%" SCNu8, &det->mask[t]) ;
		if (det->mask[t] < 1)
			det->rel_num_pix++ ;
		mean_pol += det->pixels[t*4 + 3] ;
	}
	
	mean_pol /= det->num_pix ;
	for (t = 0 ; t < det->num_pix ; ++t)
		det->pixels[t*4 + 3] /= mean_pol ;
	
	fclose(fp) ;
	
	return 0 ;
}


