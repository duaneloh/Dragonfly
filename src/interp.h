#ifndef INTERP_H
#define INTERP_H

#include "detector.h"

extern int size, center ;

void slice_gen(double*, double, double*, double*, struct detector*) ;
void slice_merge(double*, double*, double*, double*, struct detector*) ;
void symmetrize_icosahedral(double*, int) ;

#endif //INTERP_H
