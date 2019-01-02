#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "../../src/dataset.h"
#include "../../src/detector.h"

int parse_arguments(int argc, char *argv[], char *config_fname, char *powder_fname) {
	int c ;
	extern char *optarg ;
	extern int optind ;
	
	while (optind < argc) {
		if ((c = getopt(argc, argv, "hc:p:")) != -1) {
			switch (c) {
				case 'p':
					strcpy(powder_fname, optarg) ;
					break ;
				case 'c':
					strcpy(config_fname, optarg) ;
					break ;
				case 'h':
					fprintf(stderr, "Format: %s [-c config_fname] [-p powder_fname]\n", argv[0]) ;
					fprintf(stderr, "Default: -c config.ini -p data/powder.bin\n") ;
					return 1 ;
			}
		}
	}
	
	return 0 ;
}

char* remove_ext(char *fullName) {
	char *out = malloc(500 * sizeof(char)) ;
	strcpy(out,fullName) ;
	if (strrchr(out,'.') != NULL)
		*strrchr(out,'.') = 0 ;
	return out ;
}

int main(int argc, char *argv[]) {
	struct detector *det ;
	struct dataset *curr, *frames ;
	char config_fname[1024], powder_fname[1024] ;
	double *powder ;
	long t ;
	FILE *fp ;
	
	strcpy(config_fname, "config.ini") ;
	strcpy(powder_fname, "data/powder.bin") ;
	if (parse_arguments(argc, argv, config_fname, powder_fname))
		return 1 ;
	
	generate_detectors(config_fname, "emc", &det, 1) ;
	frames = malloc(sizeof(struct dataset)) ;
	generate_data(config_fname, "in", "emc", det, frames) ;
	
	powder = malloc(det->num_pix * sizeof(double)) ;
	fp = fopen(powder_fname, "rb") ;
	fread(powder, sizeof(double), det->num_pix, fp) ;
	fclose(fp) ;
	
	for (t = 0 ; t < det->num_pix ; ++t)
		powder[t] /= frames->tot_num_data ;
	
	curr = frames ;
	while (curr != NULL) {
		for (t = 0 ; t < curr->multi_total ; ++t) {
			if (curr->count_multi[t] > 1 && powder[curr->place_multi[t]] < 0.0224)
				curr->count_multi[t] = 0 ;
		}
		
		sprintf(curr->filename, "%s-noc.emc", remove_ext(curr->filename)) ;
		fprintf(stderr, "Writing to %s\n", curr->filename) ;
		write_dataset(curr) ;
		
		curr = curr->next ;
	}
	
	free_detector(det) ;
	free_data(0, frames) ;
	free(powder) ;
	
	return 0 ;
}

