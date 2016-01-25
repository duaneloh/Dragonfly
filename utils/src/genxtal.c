#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DAMP 5

double *object ;
long size ;

double gaussian(double, double) ;
void interpolate(double, double, double, double) ;

int main(int argc, char *argv[]) {
	FILE *fp ;
	long i[3], x, y, z ;
	long center, ksize, krad, spotnum[3], hsize ;
	long vol ; 
	double t[3], a[3] ;
	double blur, val, dist ;
	double *kernel, *indices ;
	char fname[999] ;
	
	if (argc < 5) {
		fprintf(stderr, "Format: %s <ax> <ay> <az> <blur>\n", argv[0]) ;
		return 1 ;
	}
	
	a[0] = atof(argv[1]) ; // Lattice constant in voxel units
	a[1] = atof(argv[2]) ; // Lattice constant in voxel units
	a[2] = atof(argv[3]) ; // Lattice constant in voxel units
	blur = atof(argv[4]) ; // Width of Gaussian blur of each bragg spot
	
	size = 201 ;
	center = size / 2 ;
	
	vol = (long)size*size*size ;
	fprintf(stderr, "vol = %ld = %.3f G\n", vol, vol/1024./1024./1024.) ;
	srand(time(NULL)) ;
	
	ksize = (int) 10 * blur ;
	krad = (ksize - 1) / 2 ;
	kernel = malloc(ksize * ksize * ksize * sizeof(double)) ;
	for (i[0] = 0 ; i[0] < ksize ; ++i[0])
	for (i[1] = 0 ; i[1] < ksize ; ++i[1])
	for (i[2] = 0 ; i[2] < ksize ; ++i[2])
		kernel[i[0]*ksize*ksize + i[1]*ksize + i[2]] = 
			gaussian(
				sqrt((krad-i[0])*(krad-i[0]) + 
					(krad-i[1])*(krad-i[1]) + 
					(krad-i[2])*(krad-i[2])), 
				blur) ;
	fprintf(stderr, "Generated kernel with ksize = %ld\n", ksize) ;
	
	hsize = 0 ;
	for (x = 0 ; x < 3 ; ++x) {
		spotnum[x] = (int) floor(size / a[x] / 2.) ;
		if (hsize < spotnum[x]) 
			hsize = spotnum[x] ;
	}
	fprintf(stderr, "spotnums = %ld, %ld, %ld\n", spotnum[0], spotnum[1], spotnum[2]) ;
	
	indices = calloc(hsize * hsize * hsize , sizeof(double)) ;
	object = calloc(vol, sizeof(double)) ;
	
	for (i[0] = -spotnum[0] ; i[0] <= spotnum[0] ; ++i[0])
	for (i[1] = -spotnum[1] ; i[1] <= spotnum[1] ; ++i[1])
	for (i[2] = -spotnum[2] ; i[2] <= spotnum[2] ; ++i[2]) {
		dist = 0. ;
		for (x = 0 ; x < 3 ; ++x) {
			// Reciprocal space coordinates
			t[x] = a[x] * i[x] ;
			
			// Calculate distance from origin
			dist += t[x] * t[x] ;
			
			// Center moved to (center,center,center) to make all coordinates positive.
			t[x] += center ;
		}
		
		// Exclude beamstop
		if (dist < 10.4 * 10.4)
			continue ;
		
		// Completely random peak heights
		val = 0.8 + 0.2 * ((double) rand()) / RAND_MAX ;
		
		// Take values from known intensity file
//		val = intens[abs(i[0])][abs(i[1])][abs(i[2])] * 0.0001 ;
		
		// Convolve with Gaussian spot kernel
		for (x = 0 ; x < ksize ; ++x)
		for (y = 0 ; y < ksize ; ++y)
		for (z = 0 ; z < ksize ; ++z)
			interpolate(t[0]+x-krad, t[1]+y-krad, t[2]+z-krad, val*kernel[x*ksize*ksize + y*ksize + z]) ;
	}
	fprintf(stderr, "Generated object with spotnums = %ld, %ld, %ld\n", spotnum[0], spotnum[1], spotnum[2]) ;
	
/*	// Add first order superlattice peaks
	t[0] = center ;
	t[1] = a[1] * 0.5 + center ;
	t[2] = center ;
	
	val = 0.8 + 0.2 * ((double) rand()) / RAND_MAX ;
	for (x = 0 ; x < ksize ; ++x)
	for (y = 0 ; y < ksize ; ++y)
	for (z = 0 ; z < ksize ; ++z)
		interpolate(t[0]+x-krad, t[1]+y-krad, t[2]+z-krad, val*kernel[x*ksize*ksize + y*ksize + z]) ;
		
	t[1] = -a[1] * 0.5 + center ;
	val = 0.8 + 0.2 * ((double) rand()) / RAND_MAX ;
	for (x = 0 ; x < ksize ; ++x)
	for (y = 0 ; y < ksize ; ++y)
	for (z = 0 ; z < ksize ; ++z)
		interpolate(t[0]+x-krad, t[1]+y-krad, t[2]+z-krad, val*kernel[x*ksize*ksize + y*ksize + z]) ;
*/	
	// Write to file
	if (argc > 5)
		fp = fopen(argv[5], "wb") ;
	else {
		sprintf(fname, "data/crystal_%ld.bin", size) ;
		fp = fopen(fname, "wb") ;
	}
	fwrite(object, vol, sizeof(double), fp) ;
	fclose(fp) ;
	
	// Free memory
	free(indices) ;
	free(object) ;
	free(kernel) ;
	
	return 0 ;
}


double gaussian(double x, double width) {
	return exp(- x*x / 2 / width/width) ;
}


void interpolate(double tx, double ty, double tz, double val) {
	long x, y, z, lsize = size ;
	double cx, cy, cz, fx, fy, fz ;
	
	if (tx < 0. || tx >= size - 1
	 || ty < 0. || ty >= size - 1
	 || tz < 0. || tz >= size - 1)
		return ;
	
	x = tx ;
	y = ty ;
	z = tz ;
	fx = tx - x ;
	fy = ty - y ;
	fz = tz - z ;
	cx = 1. - fx ;
	cy = 1. - fy ;
	cz = 1. - fz ;
	
	object[x*lsize*lsize + y*lsize + z] += cx * cy * cz * val ;
	object[x*lsize*lsize + y*lsize + (z+1)] += cx * cy * fz * val ;
	object[x*lsize*lsize + (y+1)*lsize + z] += cx * fy * cz * val ;
	object[x*lsize*lsize + (y+1)*lsize + (z+1)] += cx * fy * fz * val ;
	object[(x+1)*lsize*lsize + y*lsize + z] += fx * cy * cz * val ;
	object[(x+1)*lsize*lsize + y*lsize + (z+1)] += fx * cy * fz * val ;
	object[(x+1)*lsize*lsize + (y+1)*lsize + z] += fx * fy * cz * val ;
	object[(x+1)*lsize*lsize + (y+1)*lsize + (z+1)] += fx * fy * fz * val ;
}
