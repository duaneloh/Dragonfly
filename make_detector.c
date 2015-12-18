/*

Setup experimental configuration

compile:
gcc make_detector.c -O3 -lm -o det

usage:
./det L qmax sigma D

needs:
det_map.dat

makes:
detector.dat

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PI ((double) 3.14159265359)

char pix_map[128] = "det_map.dat" ;
double beam_vec[3] = { 0., 0., 1. } ;
int L, qmax ;
double sigma, D ;

int main(int argc, char *argv[]){

	if (argc == 5){
		L = atoi(argv[1]) ;
		qmax = atoi(argv[2]) ;
		sigma = atof(argv[3]) ;
		D = atof(argv[4]) ;
	}
	else{
		printf("Incorrect number of input to make_detector.c!!\n") ;
		return 0 ;
	}

	int i, j, k, Npix = 0, Nstop = 0 ;
	double x, y, sx, sy, sz, qx, qy, qz, qmin, norm, q, solid_angle ;
	
	// define the region blocked by the beamstop
	qmin = 1.43*sigma ;
	
	// the incident beam direction with respect to the detector (from the source towards the sample)
	norm = sqrt(beam_vec[0]*beam_vec[0] + beam_vec[1]*beam_vec[1] + beam_vec[2]*beam_vec[2]) ;
	D *= norm/beam_vec[2] ;
	beam_vec[0] *= D/norm ;
	beam_vec[1] *= D/norm ;
	beam_vec[2] *= D/norm ;

	// count Npix
	for (i = 0 ; i < 2*L+1 ; i++){
		for (j = 0 ; j < 2*L+1 ; j++){

			x = i - L ;
			y = j - L ;

			if (x*x + y*y > L*L)
				continue ;

			sx = beam_vec[0] + x ;
			sy = beam_vec[1] + y ;
			sz = beam_vec[2] ;
			norm = sqrt(sx*sx + sy*sy + sz*sz) ;
			sx *= D/norm ;
			sy *= D/norm ;
			sz *= D/norm ;
			qx = sx - beam_vec[0] ;
			qy = sy - beam_vec[1] ;
			qz = sz - beam_vec[2] ;
			
			if (qx*qx + qy*qy + qz*qz <= qmin*qmin)
				continue ;

			Npix += 1 ;
		}
	}

	// count Nstop
	for (i = 0 ; i < 2*qmax+1 ; i++){
		for (j = 0 ; j < 2*qmax+1 ; j++){
			for (k = 0 ; k < 2*qmax+1 ; k++){
				qx = i - qmax ;
				qy = j - qmax ;
				qz = k - qmax ;
				if (qx*qx + qy*qy + qz*qz < qmin*qmin)
					Nstop += 1 ;
			}
		}
	}

	FILE *outFp = fopen("detector.dat", "w") ;
	fprintf(outFp, "%d %d %d\n", qmax, Npix, Nstop);

	for (i = 0 ; i < 2*L+1 ; i++){
		for (j = 0 ; j < 2*L+1 ; j++){

			x = i - L ;
			y = j - L ;

			if (x*x + y*y > L*L)
				continue ;

			sx = beam_vec[0] + x ;
			sy = beam_vec[1] + y ;
			sz = beam_vec[2] ;
			norm = sqrt(sx*sx + sy*sy + sz*sz) ;
			sx *= D/norm ;
			sy *= D/norm ;
			sz *= D/norm ;
			qx = sx - beam_vec[0] ;
			qy = sy - beam_vec[1] ;
			qz = sz - beam_vec[2] ;
			
			if (qx*qx + qy*qy + qz*qz <= qmin*qmin)
				continue ;

			solid_angle = 1*1*D/pow(sqrt(x*x + y*y + D*D), 3) ;

			fprintf(outFp, "%lf\n", qx);
			fprintf(outFp, "%lf\n", qy);
			fprintf(outFp, "%lf\n", qz);
			fprintf(outFp, "%21.15e\n", solid_angle) ;
		}
	}

	for (i = 0 ; i < 2*qmax+1 ; i++){
		for (j = 0 ; j < 2*qmax+1 ; j++){
			for (k = 0 ; k < 2*qmax+1 ; k++){
				qx = i - qmax ;
				qy = j - qmax ;
				qz = k - qmax ;
				if (qx*qx + qy*qy + qz*qz < qmin*qmin){
					fprintf(outFp, "%d\n", i - qmax);
					fprintf(outFp, "%d\n", j - qmax);
					fprintf(outFp, "%d\n", k - qmax);
				}
			}
		}
	}
	
	fclose(outFp) ;

	return 0 ;
}
