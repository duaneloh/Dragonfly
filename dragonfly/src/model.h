#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include "detector.h"

enum model_type{MODEL_3D, MODEL_2D, MODEL_RZ} ;

struct model {
	enum model_type mtype ;
    long size, center, vol ;
    int num_modes ;
    double *model1, *model2, *inter_weight ;
} ;

void make_rot_quat(double*, double[3][3]) ;
void make_rot_angle(double, double[2][2]) ;
void slice_gen3d(double*, int, double*, struct detector*, struct model*) ;
void slice_gen2d(double*, int, double*, struct detector*, struct model*) ;
void slice_genrz(double*, int, double*, struct detector*, struct model*) ;
void slice_merge3d(double*, int, double*, struct detector*, struct model*) ;
void slice_merge2d(double*, int, double*, struct detector*, struct model*) ;
void slice_mergerz(double*, int, double*, struct detector*, struct model*) ;
void rotate_model(double[3][3], double*, int, int, double*) ;
void symmetrize_icosahedral(double*, int) ;
void symmetrize_octahedral(double*, int) ;
void symmetrize_friedel(double*, int) ;
void symmetrize_friedel2d(double*, int, int) ;

#endif //MODEL_H
