#ifndef INTERP_H
#define INTERP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include "detector.h"

void make_rot_quat(double*, double[3][3]) ;
void slice_gen(double*, double, double*, double*, long, struct detector*) ;
void slice_merge(double*, double*, double*, double*, long, struct detector*) ;
void rotate_model(double[3][3], double*, int, double*) ;
void symmetrize_icosahedral(double*, int) ;
void symmetrize_friedel(double*, int) ;

#endif //INTERP_H
