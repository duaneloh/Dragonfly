#ifndef QUAT_H
#define QUAT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>

struct rotation {
	int num_rot, num_rot_p ;
	double *quat ;
	int icosahedral_flag ;
} ;

extern char config_section[1024] ;

int quat_gen(int, struct rotation*) ;
int parse_quat(char*, struct rotation*) ;
void divide_quat(int, int, struct rotation*) ;
void free_quat(struct rotation*) ;
int generate_quaternion(char*, struct rotation*) ;

#endif //QUAT_H
