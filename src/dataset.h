#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "detector.h"

struct dataset {
	int num_data, num_pix ;
	long ones_total, multi_total ;
	double mean_count ;
	int *ones, *multi, *place_ones, *place_multi, *count_multi ;
	char filename[999] ;

	// Pointer to next dataset
	struct dataset *next ;
	
	// Need only be defined for head dataset
	int tot_num_data ;
	double tot_mean_count ;
	int *count ;
} ;

int parse_dataset(char*, struct detector*, struct dataset*) ;
int parse_data(char*, struct detector*, struct dataset*) ;
void free_data(int, struct dataset*) ;

#endif //DATASET_H
