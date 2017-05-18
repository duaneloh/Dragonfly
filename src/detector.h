#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdint.h>

struct detector {
	int num_pix, rel_num_pix ;
	double *pixels ;
	uint8_t *mask ;
} ;

int parse_detector(char*, struct detector**) ;

#endif //DETECTOR_H
