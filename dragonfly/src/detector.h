#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdint.h>

struct detector {
	char *fname ;
	int num_pix ;
	double *qvals, *corr ;
	uint8_t *raw_mask ;
	double *background ;

	double detd, ewald_rad ;
} ;

#endif // DETECTOR_H
