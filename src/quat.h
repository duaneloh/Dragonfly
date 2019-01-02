#ifndef QUAT_H
#define QUAT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>

struct rotation {
	int num_rot, num_rot_p ;
	double *quat ;
	int icosahedral_flag ;
} ;

int quat_gen(int, struct rotation*) ;
int parse_quat(char*, struct rotation*) ;
void divide_quat(int, int, int, struct rotation*) ;
void free_quat(struct rotation*) ;
int generate_quaternion(char*, char*, struct rotation*) ;

#endif //QUAT_H
