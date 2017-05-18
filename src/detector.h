#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdint.h>

struct detector {
	int num_pix, rel_num_pix ;
	double *pixels ;
	uint8_t *mask ;
} ;

int parse_detector(char*, struct detector*) ;
void free_detector(struct detector*) ;

#endif //DETECTOR_H
