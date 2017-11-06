#ifndef EMC_H
#define EMC_H

#include "detector.h"
#include "dataset.h"
#include "interp.h"
#include "quat.h"
#include "params.h"
#include "iterate.h"

#define PROB_MIN 0.000001

struct detector *det ;
struct rotation *quat ;
struct dataset *frames, *merge_frames ;
struct iterate *iter ;
struct params param ;

int rank, num_proc ;

// setup.c
int setup(char*, int) ;
void free_mem() ;

// max.c
double maximize() ;

#endif
