#ifndef DATASET_H
#define DATASET_H

struct dataset {
	int num_data, tot_num_data, num_pix ;
	long ones_total, multi_total ;
	double mean_count, tot_mean_count ;
	int *ones, *multi, *place_ones, *place_multi, *count_multi ;
	struct dataset *next ;
	char filename[999] ;
} ;

int parse_dataset(char*, struct detector*, struct dataset*) ;
int parse_data(char*, struct detector*, struct dataset*) ;

#endif //DATASET_H
