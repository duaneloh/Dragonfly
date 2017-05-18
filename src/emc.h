#ifndef EMC_H
#define EMC_H

#include <stdint.h>
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

int *count ;
double *sum_fact ;
uint8_t *blacklist ;
int num_blacklist ;

// setup.c
int setup(char*, int) ;
void free_mem() ;

// max.c
double maximize() ;

#endif
