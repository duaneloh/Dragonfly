#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdint.h>
#include <libgen.h>
#include <math.h>

struct detector {
	int num_pix, rel_num_pix ;
	double detd, ewald_rad ;
	double *pixels ;
	uint8_t *mask ;
	
	// Only relevant for first detector in list
	int num_det, num_dfiles, mapping[1024] ;
} ;

double generate_detectors(char*, char*, struct detector**, int) ;
double parse_detector(char*, struct detector*, int) ;
double parse_detector_list(char*, struct detector**, int) ;
void free_detector(struct detector*) ;

#endif //DETECTOR_H
