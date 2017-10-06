#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdint.h>
#include <math.h>

struct detector {
	int num_pix, rel_num_pix ;
	double detd, ewald_rad ;
	double *pixels ;
	uint8_t *mask ;
} ;

double parse_detector(char*, struct detector*, int) ;
void free_detector(struct detector*) ;

#endif //DETECTOR_H
