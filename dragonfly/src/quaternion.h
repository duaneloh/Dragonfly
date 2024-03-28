#ifndef QUAT_H
#define QUAT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>

struct quaternion {
	int num_div, num_rot, num_rot_p ;
	double *quats ;
	int reduced, icosahedral_flag, octahedral_flag ;
} ;

int quat_gen(int, struct quaternion*) ;
void voronoi_subset(struct quaternion*, struct quaternion*, int*) ;

#endif //QUAT_H
