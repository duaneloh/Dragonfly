#include <string.h>
#include <libgen.h>
#include <sys/time.h>
#include "../../src/quat.h"

int main(int argc, char* argv[]) {
	struct rotation *quat ;
	int r, num_div ;
	char config_fname[1024], quat_fname[1024] ;
	char line[1024], *token ;
	FILE *fp ;
	struct timeval t1, t2 ;
	
	gettimeofday(&t1, NULL) ;
	
	if (argc > 1)
		strncpy(config_fname, argv[1], 1023) ;
	else
		strcpy(config_fname, "config.ini") ;
	
	num_div = 0 ;
	quat_fname[0] = '\0' ;

	fp = fopen(config_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", config_fname) ;
		return 1 ;
	}
	while (fgets(line, 1024, fp) != NULL) {
		token = strtok(line, " =") ;
		if (token[0] == '#' || token[0] == '\n' || token[0] == '[')
			continue ;
		
		if (strcmp(token, "num_div") == 0)
			num_div = atoi(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "out_quat_file") == 0)
			strcpy(quat_fname, strtok(NULL, " =\n")) ;
	}
	fclose(fp) ;
	
	if (num_div == 0) {
		fprintf(stderr, "Need num_div (number of divisions of 600-cell)\n") ;
		return 1 ;
	}
	if (quat_fname[0] == '\0')
		strcpy(quat_fname, "data/quat.dat") ;
	
	strcpy(line, dirname(config_fname)) ;
	strcat(strcat(line, "/"), quat_fname) ;
	fprintf(stderr, "Output name: %s\n", line) ;

	quat = calloc(1, sizeof(struct rotation)) ;
	quat_gen(num_div, quat) ;
	
	fp = fopen(line, "w") ;
	fprintf(fp, "%d\n", quat->num_rot) ;
	for (r = 0 ; r < 5*quat->num_rot ; ++r) {
		fprintf(fp, "%+17.15f ", quat->quat[r]) ;
		if ((r+1) % 5 == 0)
			fprintf(fp, "\n") ;
	}
	fclose(fp) ;
	free_quat(quat) ;
	
	gettimeofday(&t2, NULL) ;
	fprintf(stderr, "Computation time = %f s\n", ((double) t2.tv_sec - t1.tv_sec + 1.e-6*(t2.tv_usec - t1.tv_usec))) ;
	
	return 0 ;
}
