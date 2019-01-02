#ifndef EMC_H
#define EMC_H

#include "detector.h"
#include "dataset.h"
#include "quat.h"
#include "params.h"
#include "iterate.h"
#include "interp.h"

#define PROB_MIN 0.000001

struct detector *det ;
struct rotation *quat ;
struct dataset *frames, *merge_frames ;
struct iterate *iter ;
struct params *param ;

// setup_emc.c
int setup(char*, int) ;
void free_mem(void) ;

// max_emc.c
double maximize(void) ;

// recon_emc.c
int parse_arguments(int, char**, int*, int*, char*) ;
void write_log_file_header(int) ;
void emc(void) ;
void update_model(double) ;

// interp function pointers
void (*slice_gen)(double*, double, double*, double*, long, struct detector*) ;
void (*slice_merge)(double*, double*, double*, double*, long, struct detector*) ;

#endif
