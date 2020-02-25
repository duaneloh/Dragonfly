#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "../../src/utils.h"
#include "../../src/quat.h"
#include "../../src/interp.h"

long s, c, rmax, rmin, max_r = 0 ;
double i2i2, max_corr ;

int parse_arguments(int argc, char *argv[], char *output_fname, char *intens_fname1, char *intens_fname2) {
	extern char *optarg ;
	extern int optind ;
	int chararg ;
	char usage_string[1024] ;
	
	omp_set_num_threads(omp_get_max_threads()) ;
	s = -1 ;
	rmax = -1 ;
	rmin = 2 ;
	i2i2 = 0. ;
	sprintf(usage_string, "Usage:\n%s [-s size] [-t num_threads] [-o output_name]\n\t[-R rmax] [-r rmin] [-h]\n\t<intens_fname1> <intens_fname2>\n", argv[0]) ;
	
	while (optind < argc) {
		if ((chararg = getopt(argc, argv, "r:R:s:t:o:h")) != -1) {
			switch (chararg) {
				case 't':
					omp_set_num_threads(atoi(optarg)) ;
					break ;
				case 's':
					s = atoi(optarg) ;
					c = s/2 ;
					break ;
				case 'o':
					strcpy(output_fname, optarg) ;
					break ;
				case 'R':
					rmax = atoi(optarg) ;
					break ;
				case 'r':
					rmin = atoi(optarg) ;
					break ;
				case 'h':
					fprintf(stderr, "Utility to align and compare two 3D intensity models.\n") ;
					fprintf(stderr, "The first model is rotated to match the second and radial CC is calculated\n") ;
					fprintf(stderr, "%s", usage_string) ;
					return 1 ;
			}
		}
		else {
			strcpy(intens_fname1, argv[optind++]) ;
			strcpy(intens_fname2, argv[optind++]) ;
		}
	}
	
	if (s == -1) {
		fprintf(stderr, "Need size of intensity volume (-s)\n") ;
		fprintf(stderr, "%s", usage_string) ;
		return 1 ;
	}
	if (intens_fname1[0] == '\0' || intens_fname2[0] == '\0') {
		fprintf(stderr, "Need two intensity files\n") ;
		fprintf(stderr, "%s", usage_string) ;
		return 1 ;
	}
	if (rmax == -1)
		rmax = c - 1 ;
	
	return 0 ;
}

void calc_corr(struct rotation *quat, double *m1, double *m2) {
	long vol = s*s*s ;
	max_corr = -DBL_MAX ;
	
	#pragma omp parallel default(shared)
	{
		int x, r, priv_max_r = 0 ;
		double i1i2, i1i1, corr, priv_max_corr = -DBL_MAX ;
		double rot[3][3] ;
		double *rotmodel = malloc(vol * sizeof(double)) ;
		int rank = omp_get_thread_num() ;
		
		// For each orientation
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < quat->num_rot ; ++r) {
			// Rotate model
			memset(rotmodel, 0, vol*sizeof(double)) ;
			make_rot_quat(&(quat->quat[r*5]), rot) ;
			rotate_model(rot, m1, s, rmax, rotmodel) ;
			
			// Calculate i1i1 and i1i2
			i1i1 = 0. ;
			i1i2 = 0. ;
			for (x = 0 ; x < vol ; ++x) {
				i1i1 += rotmodel[x] * rotmodel[x] ;
				i1i2 += m2[x] * rotmodel[x] ;
			}
			
			// Calculate corr and check for max_corr
			corr = i1i2 / sqrt(i1i1) / sqrt(i2i2) ;
			if (corr > priv_max_corr) {
				priv_max_corr = corr ;
				priv_max_r = r ;
			}
			
			if (rank == 0)
				fprintf(stderr, "\rFinished r = %d/%d", r, quat->num_rot) ;
		}
		
		#pragma omp critical(corr)
		{
			if (priv_max_corr > max_corr) {
				max_corr = priv_max_corr ;
				max_r = priv_max_r ;
			}
		}
		
		free(rotmodel) ;
	}
	
	printf("\nMax corr = %f for max_r = %ld\n", max_corr, max_r) ;
	printf("Orientation for max corr = %ld: %.9f %.9f %.9f %.9f\n", 
	       max_r, quat->quat[max_r*5], quat->quat[max_r*5+1], quat->quat[max_r*5+2], quat->quat[max_r*5+3]) ;
}

void calc_radial_corr(struct rotation *quat, double *m1, double *m2, char *fname) {
	long x, y, z, bin, vol = s*s*s ;
	double *i1i2r, *i1i1r, *i2i2r, *corr ;
	double dist, rot[3][3] ;
	double *rotmodel = calloc(vol, sizeof(double)) ;
	FILE *fp ;
	
	// Calculate rotated model
	make_rot_quat(&quat->quat[max_r*5], rot) ;
	rotate_model(rot, m1, s, rmax, rotmodel) ;
	
	// Calculate radial i1i1r, i1i2r and i2i2r
	i1i1r = calloc(c, sizeof(double)) ;
	i1i2r = calloc(c, sizeof(double)) ;
	i2i2r = calloc(c, sizeof(double)) ;
	corr = malloc(c * sizeof(double)) ;
	
	for (x = 0 ; x < s ; ++x)
	for (y = 0 ; y < s ; ++y)
	for (z = 0 ; z < s ; ++z) {
		dist = sqrt((x-c)*(x-c) + (y-c)*(y-c) + (z-c)*(z-c)) ;
		bin = (int) dist ;
		if (bin > c - 1)
			continue ;
		
		i1i1r[bin] += rotmodel[x*s*s + y*s + z] * rotmodel[x*s*s + y*s + z] ;
		i1i2r[bin] += rotmodel[x*s*s + y*s + z] * m2[x*s*s + y*s + z] ;
		i2i2r[bin] += m2[x*s*s + y*s + z] * m2[x*s*s + y*s + z] ;
	}
	
	// Calculate radial_corr
	for (bin = 0 ; bin < c ; ++bin) {
		if (i1i1r[bin] > 0. && i2i2r[bin] > 0.)
			corr[bin] = i1i2r[bin] / sqrt(i1i1r[bin]) / sqrt(i2i2r[bin]) ;
		else
			corr[bin] = 0. ;
	}
	
	// Write radial_corr to file
	fp = fopen(fname, "w") ;
	for (bin = 0 ; bin < c ; ++bin)
		fprintf(fp, "%.4ld\t%.6f\n", bin, corr[bin]) ;
	fclose(fp) ;
	
	free(i1i1r) ;
	free(i1i2r) ;
	free(i2i2r) ;
	free(corr) ;
	free(rotmodel) ;
}

void save_rotmodel(struct rotation *quat, double *m1, char *fname) {
	long vol = s*s*s ;
	double rot[3][3] ;
	double *rotmodel = calloc(vol, sizeof(double)) ;
	char rotfname[500] ;
	FILE *fp ;
	
	// Calculate rotated model
	make_rot_quat(&quat->quat[max_r*5], rot) ;
	rotate_model(rot, m1, s, rmax, rotmodel) ;
	
	// Write rotmodel to file
	char *base = remove_ext(extract_fname(fname)) ;
	sprintf(rotfname, "data/%s-rot.bin", base) ;
	free(base) ;
	fp = fopen(rotfname, "wb") ;
	fwrite(rotmodel, sizeof(double), vol, fp) ;
	fclose(fp) ;
	
	free(rotmodel) ;
}

void subtract_radial_average(double *model1, double *model2, double exponent, double *model1_rad, double *model2_rad) {
	int x, y, z, bin, *count ;
	double dist, *mean1, *mean2 ;
	
	mean1 = calloc(c, sizeof(double)) ;
	mean2 = calloc(c, sizeof(double)) ;
	count = calloc(c, sizeof(int)) ;
	
	for (x = 0 ; x < s ; ++x)
	for (y = 0 ; y < s ; ++y)
	for (z = 0 ; z < s ; ++z) {
		dist = sqrt((x-c)*(x-c) + (y-c)*(y-c) + (z-c)*(z-c)) ;
		bin = (int) dist ;
		if (bin > c-1)
			continue ;
		
		mean1[bin] += pow(model1[x*s*s + y*s + z], exponent) ;
		mean2[bin] += pow(model2[x*s*s + y*s + z], exponent) ;
		count[bin]++ ;
	}
	
	for (bin = 0 ; bin < c ; ++bin) {
		mean1[bin] /= count[bin] ;
		mean2[bin] /= count[bin] ;
	}
	
	for (x = 0 ; x < s ; ++x)
	for (y = 0 ; y < s ; ++y)
	for (z = 0 ; z < s ; ++z) {
		dist = sqrt((x-c)*(x-c) + (y-c)*(y-c) + (z-c)*(z-c)) ;
		bin = (int) dist ;
		
		if (bin > rmax || bin < rmin) {
			model1_rad[x*s*s + y*s + z] = 0. ;
			model2_rad[x*s*s + y*s + z] = 0. ;
		}
		else {
			model1_rad[x*s*s + y*s + z] = pow(model1[x*s*s + y*s + z], exponent) - mean1[bin] ;
			model2_rad[x*s*s + y*s + z] = pow(model2[x*s*s + y*s + z], exponent) - mean2[bin] ;
		}
	}
	
	free(count) ;
	free(mean1) ;
	free(mean2) ;
}

void gen_subset(struct rotation *quat, int num_div, double dmax) {
	int t, r, full_num_rot ;
	double dist, max_quat[4] ;
	
	for (t = 0 ; t < 4 ; ++t)
		max_quat[t] = quat->quat[max_r*5 + t] ;
	free(quat->quat) ;
	full_num_rot = quat_gen(num_div, quat) ;
	
	quat->num_rot = 0 ;
	for (r = 0 ; r < full_num_rot ; ++r) {
		dist = 0. ;
		for (t = 0 ; t < 4 ; ++t)
			dist += (quat->quat[r*5 + t] - max_quat[t]) * (quat->quat[r*5 + t] - max_quat[t]) ;
		
		if (dist < dmax) {
			for (t = 0 ; t < 5 ; ++t)
				quat->quat[quat->num_rot*5 + t] = quat->quat[r*5 + t] ;
			quat->num_rot++ ;
		}
	}
}

int main(int argc, char *argv[]) {
	long vol, t ;
	double *model1, *model2, *model1_rad, *model2_rad ;
	struct rotation *quat ;
	FILE *fp ;
	char intens_fname1[1024] = {'\0'}, intens_fname2[1024] = {'\0'} ;
	char output_fname[1024] = {'\0'} ;
	
	if (parse_arguments(argc, argv, output_fname, intens_fname1, intens_fname2))
		return 1 ;
	vol = s*s*s ;
	
	// Parse models
	fp = fopen(intens_fname1, "rb") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to open first file: %s\n", intens_fname1) ;
		return 1 ;
	}
	model1 = malloc(vol * sizeof(double)) ;
	fread(model1, sizeof(double), vol, fp) ;
	fclose(fp) ;
	
	fp = fopen(intens_fname2, "rb") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to open second file: %s\n", intens_fname2) ;
		free(model1) ;
		return 1 ;
	}
	model2 = malloc(vol * sizeof(double)) ;
	fread(model2, sizeof(double), vol, fp) ;
	fclose(fp) ;
	fprintf(stderr, "Parsed models from %s and %s\n", intens_fname1, intens_fname2) ;
	
	// Radial average subtraction
	model1_rad = malloc(vol * sizeof(double)) ;
	model2_rad = malloc(vol * sizeof(double)) ;
	subtract_radial_average(model1, model2, 1., model1_rad, model2_rad) ;
	fprintf(stderr, "Radial average subtracted\n") ;
	
	// Calculate i2i2 as it is not being rotated
	for (t = 0 ; t < vol ; ++t)
		i2i2 += model2_rad[t] * model2_rad[t] ;
	
	// Generate quaternion and calculate max_corr
	quat = calloc(1, sizeof(struct rotation)) ;
	quat_gen(4, quat) ;
	calc_corr(quat, model1_rad, model2_rad) ;
	
	// Generate subset and recalculate max_corr
	gen_subset(quat, 30, 0.04) ;
	calc_corr(quat, model1_rad, model2_rad) ;
	gen_subset(quat, 40, 0.01) ;
	calc_corr(quat, model1_rad, model2_rad) ;
	
	// Calculate radial_corr for best orientation and save rotated model
	char fname[500] ;
	if (output_fname == '\0') {
		char *base = remove_ext(extract_fname(intens_fname1)) ;
		sprintf(fname, "data/%s.dat", base) ;
		free(base) ;
	}
	else {
		sprintf(fname, "data/%s.dat", output_fname) ;
	}
	fprintf(stderr, "Saving FSC to %s\n", fname) ;
	
	rmin = 2 ;
	rmax = c - 1 ;
	subtract_radial_average(model1, model2, 1., model1_rad, model2_rad) ;
	calc_radial_corr(quat, model1_rad, model2_rad, fname) ;
	// Save un-(radially subtracted) rotated model
	save_rotmodel(quat, model1, fname) ;
	
	free(model1) ;
	free(model2) ;
	free(model1_rad) ;
	free(model2_rad) ;
	free(quat->quat) ;
	free(quat) ;
	
	return 0 ;
}

