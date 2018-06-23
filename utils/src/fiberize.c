#include "../../src/detector.h"
#include "../../src/interp.h"

int size, center, num_pix ;

void make_flat_detector(struct detector *det) {
	int x, y, t ;
	
	det->num_pix = size*size ;
	det->pixels = malloc(det->num_pix * 4 * sizeof(double)) ;
	det->mask = malloc(det->num_pix * sizeof(uint8_t)) ;
	for (x = 0 ; x < size ; ++x)
	for (y = 0 ; y < size ; ++y) {
		t = x*size + y ;
		det->pixels[t*4 + 0] = x - center ;
		det->pixels[t*4 + 1] = y - center ;
		det->pixels[t*4 + 2] = 0. ;
		det->pixels[t*4 + 3] = 1. ;
		det->mask[t] = 0 ;
	}
}

void rotate_slice(double *in, double angle, double weight, double *out) {
	int x, y, ix, iy ;
	double tx, ty, fx, fy, cx, cy ;	
	double rot[2][2] = {{cos(angle), -sin(angle)}, {sin(angle), cos(angle)}} ;
	
	for (x = 0 ; x < size ; ++x)
	for (y = 0 ; y < size ; ++y) {
		tx = (x-center)*rot[0][0] + (y-center)*rot[0][1] ;
		ty = (x-center)*rot[1][0] + (y-center)*rot[1][1] ;
		if (sqrt(tx*tx + ty*ty) > center - 20)
			continue ;
		
		tx += center ;
		ty += center ;
		ix = tx ;
		iy = ty ;
		fx = tx - ix ;
		fy = ty - iy ;
		cx = 1. - fx ;
		cy = 1. - fy ;
		
		out[x*size + y] += weight*(cx*cy*in[ix*size + iy] +
		                           cx*fy*in[ix*size + (iy+1)] +
		                           fx*cy*in[(ix+1)*size + iy] +
		                           fx*fy*in[(ix+1)*size + (iy+1)]) ;
	}
}

int main(int argc, char *argv[]) {
	int t, num_angles = 720 ;
	double a, c, s, w, sigma, quat[4] ;
	double *volume, *slice, *temp_slice ;
	FILE *fp ;
	struct detector *det ;
	
	if (argc < 4) {
		fprintf(stderr, "Format: %s <vol_fname> <size> <blur_sigma>\n", argv[0]) ;
		fprintf(stderr, "\t<blur_sigma> is in degrees\n") ;
		return 1 ;
	}
	size = atoi(argv[2]) ;
	sigma = atof(argv[3]) * M_PI / 180. ;
	num_pix = size*size ;
	center = size / 2 ;
	
	volume = malloc(size*size*size*sizeof(double)) ;
	fp = fopen(argv[1], "rb") ;
	fread(volume, sizeof(double), size*size*size, fp) ;
	fclose(fp) ;
	
	det = malloc(sizeof(struct detector)) ;
	make_flat_detector(det) ;
	
	slice = calloc(num_pix, sizeof(double)) ;
	temp_slice = malloc(num_pix * sizeof(double)) ;
	
	// Calculate average slice
	for (a = 0 ; a < 2.*M_PI ; a += 2.*M_PI/num_angles) {
		c = cos(a) ;
		s = sin(a) ;
		if (c > 0.) {
			quat[0] = sqrt(1 + c) / 2. ;
			quat[1] = s / sqrt(1 + c) / 2. ;
		}
		else {
			quat[0] = s / sqrt(1 - c) / 2. ;
			quat[1] = sqrt(1 - c) / 2. ;
		}
		quat[2] = quat[0] ;
		quat[3] = quat[1] ;
		
		slice_gen3d(quat, 0., temp_slice, volume, size, det) ;
		for (t = 0 ; t < num_pix ; ++t)
			slice[t] += temp_slice[t] ;
		fprintf(stderr, "\r%.2f", a*180./M_PI) ;
	}
	fprintf(stderr, "\n") ;
	for (t = 0 ; t < num_pix ; ++t) {
		slice[t] /= num_angles ;
		temp_slice[t] = 0. ;
	}
	
	// Calculate rotationally blurred slice
	num_angles = 101 ;
	for (a = -4.*sigma ; a < 4.*sigma ; a += 4.*sigma/num_angles) {
		w = exp(-a*a/2./sigma/sigma) ;
		rotate_slice(slice, a, w, temp_slice) ;
		fprintf(stderr, "\r%.2f", a*180/M_PI) ;
	}
	fprintf(stderr, "\n") ;
	
	// Save to file
	fp = fopen("data/cylindrical_slice.bin", "wb") ;
	fwrite(slice, sizeof(double), num_pix, fp) ;
	fclose(fp) ;
	
	fp = fopen("data/blurred_slice.bin", "wb") ;
	fwrite(temp_slice, sizeof(double), num_pix, fp) ;
	fclose(fp) ;
	
	free(slice) ;
	free(temp_slice) ;
	free_detector(det) ;
	
	return 0 ;
}
