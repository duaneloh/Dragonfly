/*
Generates quaternion file required by recon
	
	** DON'T RUN THIS CODE. RUN quat.sh INSTEAD. **
	
	List of quaternions representing samples of the rotation group. Each line
	has five entries, the first four being the unit quaternion and the fifth
	is a weight factor. This is required because each quaternion does not have
	the same sized Voronoi cell on the surface of the 3-sphere. Finer samples
	are generated by interpolating vertices, edges, faces and cells of the
	high symmetry 600-cell.
	
	For usage, run without command line arguments.
	
	Requires:
		n - Integer representing interpolation level. The average separation
			between nearest neighbour orientations is approximately (0.94/n)
			radians. The number of samples is 10(n+5n^3).
		vertices.dat, edges.dat, faces.dat, cells.dat - Quaternions for n=1.
			All files need to be in the aux directory. 
	
	Generates:
		quat_<n>.dat - where <n> is a two-digit version of n (by quat.sh).
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359

int check_include(double q[4]) {
	int d ;
	for (d = 0 ; d < 4 ; ++d) {
		if (q[d] > 0. && fabs(q[d]) > 1e-10)
			return 0 ;
		else if (q[d] < 0. && fabs(q[d]) > 1e-10)
			return 1 ;
		
		if (d == 3) {
			fprintf(stderr, "quat = {0,0,0,0}\n") ;
			return 0 ;
		}
	}
	
	return 0 ;
}
		

double calc_weight(double q[4], double *c) {
	int d ;
	double weight = 0., norm_q = 0., norm_c = 0. ;
	for (d = 0 ; d < 4 ; ++d) {
		weight += q[d] * c[d] ;
		norm_q += q[d] * q[d] ;
		norm_c += c[d] * c[d] ;
	}
	
	return weight / (sqrt(norm_c) * pow(norm_q, 2.)) ;
}


int main(int argc, char *argv[]) {
	int n, r, i, j, k, d, counter, num_rot ;
	double dihedral, f1, w0, norm, temp[4], *quat ;
	double *c ;
	FILE *fp ;
	char fname[500] ;
	
	if (argc < 2) {
		fprintf(stderr, "Format: %s <n>\n", argv[0]) ;
		return 1 ;
	}
	n = atoi(argv[1]) ;
	num_rot = 10 * (n + 5*n*n*n) ;
	
	quat = malloc(3 * num_rot * 5 * sizeof(double)) ;
	
	double vertices[120][4] ;
	fp = fopen("aux/vertices.dat", "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to find vertices.dat\n") ;
		return 2 ;
	}
	for (i = 0 ; i < 120 ; ++i)
	for (d = 0 ; d < 4 ; ++d)
		fscanf(fp, "%lf", &vertices[i][d]) ;
	fclose(fp) ;
	
	double edges[720][12] ;
	fp = fopen("aux/edges.dat", "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to find edges.dat\n") ;
		return 2 ;
	}
	for (i = 0 ; i < 720 ; ++i)
	for (d = 0 ; d < 12 ; ++d)
		fscanf(fp, "%lf ", &edges[i][d]) ;
	fclose(fp) ;
	
	double faces[1200][16] ;
	fp = fopen("aux/faces.dat", "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to find faces.dat\n") ;
		return 2 ;
	}
	for (i = 0 ; i < 1200 ; ++i)
	for (d = 0 ; d < 16 ; ++d)
		fscanf(fp, "%lf", &faces[i][d]) ;
	fclose(fp) ;
	
	double cells[600][20] ;
	fp = fopen("aux/cells.dat", "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to find cells.dat\n") ;
		return 2 ;
	}
	for (i = 0 ; i < 600 ; ++i)
	for (d = 0 ; d < 20 ; ++d)
		fscanf(fp, "%lf", &cells[i][d]) ;
	fclose(fp) ;
	
	dihedral = acos(1. / 3.) ;
	f1 = 5 * dihedral / (2 * PI) ;
	
	w0 = 0.8121328281052886 ;
	
	counter = 0 ;
	for (r = 0 ; r < 120 ; ++r) {
		if (check_include(vertices[r]) == 1) {
			for (d = 0 ; d < 4 ; ++d)
				quat[counter*5 + d] = vertices[r][d] ;
			quat[counter*5 + 4] = w0 ;
			counter++ ;
		}
	}
	fprintf(stderr, "counter = %d after vertices\n", counter) ;
	
	for (r = 0 ; r < 720 ; ++r)
	for (i = 0 ; i < n - 1 ; ++i) {
		for (d = 0 ; d < 4 ; ++d)
			temp[d] = edges[r][d] + (1. + i) / n * (edges[r][d+4] - edges[r][d]) ;
		
		if (check_include(temp) == 1) {
			norm = 0. ;
			for (d = 0 ; d < 4 ; ++d)
				norm += temp[d] * temp[d] ;
			norm = 1. / sqrt(norm) ;
			
			for (d = 0 ; d < 4 ; ++d)
				quat[counter*5 + d] = temp[d] * norm ;
			
			c = &edges[r][8] ;
			quat[counter*5 + 4] = f1 * calc_weight(temp, c) ;
			
			counter++ ;
		}
	}
	fprintf(stderr, "counter = %d after edges\n", counter) ;
	
	for (r = 0 ; r < 1200 ; ++r) 
	for (i = 0 ; i < n - 2 ; ++i)
	for (j = 0 ; j < n - 2 - i ; ++j) {
		for (d = 0 ; d < 4 ; ++d)
			temp[d] = faces[r][d] + 
					(1. + i) / n * (faces[r][d+4] - faces[r][d]) +
					(1. + j) / n * (faces[r][d+8] - faces[r][d]) ;
		
		if (check_include(temp) == 1) {
			norm = 0. ;
			for (d = 0 ; d < 4 ; ++d)
				norm += temp[d] * temp[d] ;
			norm = 1. / sqrt(norm) ;
			
			for (d = 0 ; d < 4 ; ++d)
				quat[counter*5 + d] = temp[d] * norm ;
			
			c = &faces[r][12] ;
			quat[counter*5 + 4] = calc_weight(temp, c) ;
			
			counter++ ;
		}
	}
	fprintf(stderr, "counter = %d after faces\n", counter) ;
	
	for (r = 0 ; r < 600 ; ++r)
	for (i = 0 ; i < n - 3 ; ++i)
	for (j = 0 ; j < n - 3 - i ; ++j)
	for (k = 0 ; k < n - 3 - i - j ; ++k) {
		for (d = 0 ; d < 4 ; ++d)
			temp[d] = cells[r][d] + 
					(1. + i) / n * (cells[r][d+4] - cells[r][d]) +
					(1. + j) / n * (cells[r][d+8] - cells[r][d]) +
					(1. + k) / n * (cells[r][d+12] - cells[r][d]) ;
		
		if (check_include(temp) == 1) {
			norm = 0. ;
			for (d = 0 ; d < 4 ; ++d)
				norm += temp[d] * temp[d] ;
			norm = 1. / sqrt(norm) ;
			
			for (d = 0 ; d < 4 ; ++d)
				quat[counter*5 + d] = temp[d] * norm ;
			
			c = &cells[r][16] ;
			quat[counter*5 + 4] = calc_weight(temp, c) ;
			
			counter++ ;
		}
	}
	fprintf(stderr, "counter = %d after cells\n", counter) ;
	
	sprintf(fname, "quat_%.2d_unsorted.dat", n) ;
	fp = fopen(fname, "w") ;
	
	fprintf(fp, "%d \n\n", counter) ;
	
	for (r = 0 ; r < counter ; ++r) {
		for (d = 0 ; d < 5 ; ++d)
			fprintf(fp, "%.16f\t", quat[r*5 + d]) ;
		fprintf(fp, "\n") ;
	}
	fclose(fp) ;
	
	free(quat) ;
		
	return 0 ;
}
