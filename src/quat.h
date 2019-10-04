#ifndef QUAT_H
#define QUAT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>
#include "utils.h"

struct rotation {
	int num_rot, num_rot_p ;
	double *quat, (*sym_quat)[4] ;
	int icosahedral_flag, octahedral_flag ;
} ;

int quat_gen(int, struct rotation*) ;
int parse_quat(char*, int, struct rotation*) ;
void divide_quat(int, int, int, int, struct rotation*) ;
void free_quat(struct rotation*) ;
int quat_from_config(char*, char*, struct rotation*) ;
void voronoi_subset(struct rotation*, struct rotation*, int*) ;

#endif //QUAT_H
