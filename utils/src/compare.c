#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "../../src/quat.c"

int num_rot, s, c, rmax, rmin, max_r = 0 ;
double i2i2, max_corr = -DBL_MAX ;

char* extract_fname(char* fullName) {
	return 
		strrchr(fullName,'/') != NULL
			? strrchr(fullName,'/') + 1
			: fullName ;
}

char* remove_ext(char *fullName) {
	char *out = malloc(500 * sizeof(char)) ;
	strcpy(out,fullName) ;
	if (strrchr(out,'.') != NULL)
		*strrchr(out,'.') = 0 ;
	return out ;
}

void make_rot_quat(double *quaternion, double rot[3][3]) {
	double q0, q1, q2, q3, q01, q02, q03, q11, q12, q13, q22, q23, q33 ;
	
	q0 = quaternion[0] ;
	q1 = quaternion[1] ;
	q2 = quaternion[2] ;
	q3 = quaternion[3] ;
	
	q01 = q0*q1 ;
	q02 = q0*q2 ;
	q03 = q0*q3 ;
	q11 = q1*q1 ;
	q12 = q1*q2 ;
	q13 = q1*q3 ;
	q22 = q2*q2 ;
	q23 = q2*q3 ;
	q33 = q3*q3 ;
	
	rot[0][0] = (1. - 2.*(q22 + q33)) ;
	rot[0][1] = 2.*(q12 + q03) ;
	rot[0][2] = 2.*(q13 - q02) ;
	rot[1][0] = 2.*(q12 - q03) ;
	rot[1][1] = (1. - 2.*(q11 + q33)) ;
	rot[1][2] = 2.*(q01 + q23) ;
	rot[2][0] = 2.*(q02 + q13) ;
	rot[2][1] = 2.*(q23 - q01) ;
	rot[2][2] = (1. - 2.*(q11 + q22)) ;
}

// Calculate corr using radial-average-subtracted models
void calc_corr(double *quat, double *m1, double *m2) {
	int vol = s*s*s ;
	
	#pragma omp parallel default(shared)
	{
		int x, y, z, r, i, j, priv_max_r = 0, vox[3] ;
		double fx, fy, fz, cx, cy, cz ;
		double i1i2, i1i1, corr, priv_max_corr = -DBL_MAX ;
		double rot[3][3], rot_vox[3] ;
		double *rotmodel = malloc(vol * sizeof(double)) ;
		int rank = omp_get_thread_num() ;
		
		// For each orientation
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < num_rot ; ++r) {
			// Calculate rotation matrix
			make_rot_quat(&quat[r*5], rot) ;
			
			// Zero rotated model
			for (x = 0 ; x < vol ; ++x)
				rotmodel[x] = 0. ;
			
			// Calculate rotated model
			for (vox[0] = -c ; vox[0] < s-c-1 ; ++vox[0])
			for (vox[1] = -c ; vox[1] < s-c-1 ; ++vox[1])
			for (vox[2] = -c ; vox[2] < s-c-1 ; ++vox[2]) {
				for (i = 0 ; i < 3 ; ++i) {
					rot_vox[i] = 0. ;
					for (j = 0 ; j < 3 ; ++j) 
						rot_vox[i] += rot[i][j] * vox[j] ;
					rot_vox[i] += c ;
				}
				
				if (rot_vox[0] < 0 || rot_vox[0] >= s - 1) continue ;
				if (rot_vox[1] < 0 || rot_vox[1] >= s - 1) continue ;
				if (rot_vox[2] < 0 || rot_vox[2] >= s - 1) continue ;
				
				x = (int) rot_vox[0] ;
				y = (int) rot_vox[1] ;
				z = (int) rot_vox[2] ;
				fx = rot_vox[0] - x ;
				fy = rot_vox[1] - y ;
				fz = rot_vox[2] - z ;
				cx = 1. - fx ;
				cy = 1. - fy ;
				cz = 1. - fz ;
				
/*				w = m1[(vox[0]+c)*s*s + (vox[1]+c)*s + (vox[2]+c)] ;
				
				rotmodel[x*s*s + y*s + z] += cx * cy * cz * w ;
				rotmodel[x*s*s + y*s + ((z+1)%s)] += cx * cy * fz * w ;
				rotmodel[x*s*s + ((y+1)%s)*s + z] += cx * fy * cz * w ;
				rotmodel[x*s*s + ((y+1)%s)*s + ((z+1)%s)] += cx * fy * fz * w ;
				rotmodel[((x+1)%s)*s*s + y*s + z] += fx * cy * cz * w ;
				rotmodel[((x+1)%s)*s*s + y*s + ((z+1)%s)] += fx * cy * fz * w ;
				rotmodel[((x+1)%s)*s*s + ((y+1)%s)*s + z] += fx * fy * cz * w ;
				rotmodel[((x+1)%s)*s*s + ((y+1)%s)*s + ((z+1)%s)] += fx * fy * fz * w ;
*/				
				rotmodel[(vox[0]+c)*s*s + (vox[1]+c)*s + (vox[2]+c)] =
					cx*cy*cz*m1[x*s*s + y*s + z] + 
					cx*cy*fz*m1[x*s*s + y*s + ((z+1)%s)] + 
					cx*fy*cz*m1[x*s*s + ((y+1)%s)*s + z] + 
					cx*fy*fz*m1[x*s*s + ((y+1)%s)*s + ((z+1)%s)] + 
					fx*cy*cz*m1[((x+1)%s)*s*s + y*s + z] +
					fx*cy*fz*m1[((x+1)%s)*s*s + y*s + ((z+1)%s)] +
					fx*fy*cz*m1[((x+1)%s)*s*s + ((y+1)%s)*s + z] +
					fx*fy*fz*m1[((x+1)%s)*s*s + ((y+1)%s)*s + ((z+1)%s)] ;
			}
			
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
				fprintf(stderr, "\rFinished r = %d/%d", r, num_rot) ;
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
	
	printf("\nMax corr = %f for max_r = %d\n", max_corr, max_r) ;
	printf("Orientation for max corr = %d: %.9f %.9f %.9f %.9f\n", 
	       max_r, quat[max_r*5], quat[max_r*5+1], quat[max_r*5+2], quat[max_r*5+3]) ;
}

void calc_radial_corr(double *quat, double *m1, double *m2, char *fname) {
	int x, y, z, i, j, bin, vol = s*s*s, vox[3] ;
	double fx, fy, fz, cx, cy, cz, dist ;
	double *i1i2r, *i1i1r, *i2i2r, *corr ;
	double rot[3][3], rot_vox[3] ;
	double *rotmodel = calloc(vol, sizeof(double)) ;
	FILE *fp ;
	
	// Calculate rotation matrix
	make_rot_quat(&quat[max_r*5], rot) ;
	
	// Calculate rotated model
	for (vox[0] = -c ; vox[0] < s-c-1 ; ++vox[0])
	for (vox[1] = -c ; vox[1] < s-c-1 ; ++vox[1])
	for (vox[2] = -c ; vox[2] < s-c-1 ; ++vox[2]) {
		for (i = 0 ; i < 3 ; ++i) {
			rot_vox[i] = 0. ;
			for (j = 0 ; j < 3 ; ++j) 
				rot_vox[i] += rot[i][j] * vox[j] ;
			rot_vox[i] += c ;
		}
		
		if (rot_vox[0] < 0 || rot_vox[0] >= s - 1) continue ;
		if (rot_vox[1] < 0 || rot_vox[1] >= s - 1) continue ;
		if (rot_vox[2] < 0 || rot_vox[2] >= s - 1) continue ;
		
		x = (int) rot_vox[0] ;
		y = (int) rot_vox[1] ;
		z = (int) rot_vox[2] ;
		fx = rot_vox[0] - x ;
		fy = rot_vox[1] - y ;
		fz = rot_vox[2] - z ;
		cx = 1. - fx ;
		cy = 1. - fy ;
		cz = 1. - fz ;
		
		rotmodel[(vox[0]+c)*s*s + (vox[1]+c)*s + (vox[2]+c)] =
			cx*cy*cz*m1[x*s*s + y*s + z] + 
			cx*cy*fz*m1[x*s*s + y*s + ((z+1)%s)] + 
			cx*fy*cz*m1[x*s*s + ((y+1)%s)*s + z] + 
			cx*fy*fz*m1[x*s*s + ((y+1)%s)*s + ((z+1)%s)] + 
			fx*cy*cz*m1[((x+1)%s)*s*s + y*s + z] +
			fx*cy*fz*m1[((x+1)%s)*s*s + y*s + ((z+1)%s)] +
			fx*fy*cz*m1[((x+1)%s)*s*s + ((y+1)%s)*s + z] +
			fx*fy*fz*m1[((x+1)%s)*s*s + ((y+1)%s)*s + ((z+1)%s)] ;
	}
	
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
		fprintf(fp, "%.4d\t%.6f\n", bin, corr[bin]) ;
	fclose(fp) ;
	
	free(i1i1r) ;
	free(i1i2r) ;
	free(i2i2r) ;
	free(corr) ;
	free(rotmodel) ;
}

void save_rotmodel(double *quat, double *m1, char *fname) {
	int x, y, z, i, j, vol = s*s*s, vox[3] ;
	double fx, fy, fz, cx, cy, cz ;
	double rot[3][3], rot_vox[3] ;
	double *rotmodel = calloc(vol, sizeof(double)) ;
	char rotfname[500] ;
	FILE *fp ;
	
	// Calculate rotation matrix
	make_rot_quat(&quat[max_r*5], rot) ;
	
	// Calculate rotated model
	for (vox[0] = -c ; vox[0] < s-c-1 ; ++vox[0])
	for (vox[1] = -c ; vox[1] < s-c-1 ; ++vox[1])
	for (vox[2] = -c ; vox[2] < s-c-1 ; ++vox[2]) {
		for (i = 0 ; i < 3 ; ++i) {
			rot_vox[i] = 0. ;
			for (j = 0 ; j < 3 ; ++j) 
				rot_vox[i] += rot[i][j] * vox[j] ;
			rot_vox[i] += c ;
		}
		
		if (rot_vox[0] < 0 || rot_vox[0] >= s - 1) continue ;
		if (rot_vox[1] < 0 || rot_vox[1] >= s - 1) continue ;
		if (rot_vox[2] < 0 || rot_vox[2] >= s - 1) continue ;
		
		x = (int) rot_vox[0] ;
		y = (int) rot_vox[1] ;
		z = (int) rot_vox[2] ;
		fx = rot_vox[0] - x ;
		fy = rot_vox[1] - y ;
		fz = rot_vox[2] - z ;
		cx = 1. - fx ;
		cy = 1. - fy ;
		cz = 1. - fz ;
		
		rotmodel[(vox[0]+c)*s*s + (vox[1]+c)*s + (vox[2]+c)] =
			cx*cy*cz*m1[x*s*s + y*s + z] + 
			cx*cy*fz*m1[x*s*s + y*s + ((z+1)%s)] + 
			cx*fy*cz*m1[x*s*s + ((y+1)%s)*s + z] + 
			cx*fy*fz*m1[x*s*s + ((y+1)%s)*s + ((z+1)%s)] + 
			fx*cy*cz*m1[((x+1)%s)*s*s + y*s + z] +
			fx*cy*fz*m1[((x+1)%s)*s*s + y*s + ((z+1)%s)] +
			fx*fy*cz*m1[((x+1)%s)*s*s + ((y+1)%s)*s + z] +
			fx*fy*fz*m1[((x+1)%s)*s*s + ((y+1)%s)*s + ((z+1)%s)] ;
	}
	
	// Write rotmodel to file
	sprintf(rotfname, "data/%s-rot.bin", remove_ext(extract_fname(fname))) ;
	fp = fopen(rotfname, "wb") ;
	fwrite(rotmodel, sizeof(double), vol, fp) ;
	fclose(fp) ;
	
	free(rotmodel) ;
}

void subtract_radial_average(double *model1, double *model2, double *model1_rad, double *model2_rad) {
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
		
		mean1[bin] += model1[x*s*s + y*s + z] ;
		mean2[bin] += model2[x*s*s + y*s + z] ;
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
			model1_rad[x*s*s + y*s + z] = model1[x*s*s + y*s + z] - mean1[bin] ;
			model2_rad[x*s*s + y*s + z] = model2[x*s*s + y*s + z] - mean2[bin] ;
		}
	}
	
	free(count) ;
	free(mean1) ;
	free(mean2) ;
}

void gen_subset(double **quaternion, int num_div, double dmax) {
	int t, r, full_num_rot ;
	double dist, max_quat[4] ;
	
	for (t = 0 ; t < 4 ; ++t)
		max_quat[t] = (*quaternion)[max_r*5 + t] ;
	free(*quaternion) ;
	full_num_rot = quat_gen(num_div, quaternion) ;
	
	num_rot = 0 ;
	for (r = 0 ; r < full_num_rot ; ++r) {
		dist = 0. ;
		for (t = 0 ; t < 4 ; ++t)
			dist += ((*quaternion)[r*5 + t] - max_quat[t]) * ((*quaternion)[r*5 + t] - max_quat[t]) ;
		
		if (dist < dmax) {
			for (t = 0 ; t < 5 ; ++t)
				(*quaternion)[num_rot*5 + t] = (*quaternion)[r*5 + t] ;
			num_rot++ ;
		}
	}
	printf("\nnew num_rot = %d\n", num_rot) ;
}

int main(int argc, char *argv[]) {
	long vol, t ;
	double *model1, *model2, *quat, *model1_rad, *model2_rad ;
	FILE *fp ;
	char intens_fname1[999], intens_fname2[999] ;
	char output_fname[999] ;
	
	extern char *optarg ;
	extern int optind ;
	int chararg ;
	
	omp_set_num_threads(omp_get_max_threads()) ;
	s = -1 ;
	output_fname[0] = '\0' ;
	rmax = -1 ;
	rmin = 10 ;
	
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
					fprintf(stderr, "Format: %s \n\t[-s size]\n\t[-t num_threads]\n\t[-o output_name]\n\t[-R rmax]\n\t[-r rmin]\n\t[-h]\n", argv[0]) ;
					fprintf(stderr, "\t\t<intens_fname1> <intens_fname2>\n") ;
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
		return 1 ;
	}
	if (rmax == -1)
		rmax = c - 1 ;
	vol = s*s*s ;
	i2i2 = 0. ;
	
	// Parse models
	model1 = malloc(vol * sizeof(double)) ;
	model1_rad = malloc(vol * sizeof(double)) ;
	fp = fopen(intens_fname1, "rb") ;
	fread(model1, sizeof(double), vol, fp) ;
	fclose(fp) ;
	
	model2 = malloc(vol * sizeof(double)) ;
	model2_rad = malloc(vol * sizeof(double)) ;
	fp = fopen(intens_fname2, "rb") ;
	fread(model2, sizeof(double), vol, fp) ;
	fclose(fp) ;
	fprintf(stderr, "Parsed models\n") ;
	
	// Radial average subtraction
	subtract_radial_average(model1, model2, model1_rad, model2_rad) ;
	fprintf(stderr, "Radial average subtracted\n") ;
	
	// Calculate i2i2 as it is not being rotated
	for (t = 0 ; t < vol ; ++t)
		i2i2 += model2_rad[t] * model2_rad[t] ;
	
	// Parse quaternion and calculate max_corr
	num_rot = quat_gen(4, &quat) ;
	calc_corr(quat, model1, model2) ;
	
	// Generate subset and recalculate max_corr
	gen_subset(&quat, 30, 0.06) ;
	max_corr = -DBL_MAX ;
	calc_corr(quat, model1_rad, model2_rad) ;
	
	// Generate subset and recalculate max_corr
	gen_subset(&quat, 50, 0.02) ;
	max_corr = -DBL_MAX ;
	calc_corr(quat, model1_rad, model2_rad) ;
	
	// Calculate radial_corr for best orientation and save rotated model
	char fname[500] ;
	if (output_fname == '\0')
		sprintf(fname, "data/%s.dat", remove_ext(extract_fname(intens_fname1))) ;
	else
		sprintf(fname, "data/%s.dat", output_fname) ;
	fprintf(stderr, "Saving FSC to %s\n", fname) ;
	
	rmin = 2 ;
	rmax = c - 1 ;
	subtract_radial_average(model1, model2, model1_rad, model2_rad) ;
	calc_radial_corr(quat, model1_rad, model2_rad, fname) ;
	save_rotmodel(quat, model1, fname) ;
	
	free(model1) ;
	free(model2) ;
	free(model1_rad) ;
	free(model2_rad) ;
	free(quat) ;
	
	return 0 ;
}

