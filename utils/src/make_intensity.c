/*

Simulate intensity in k-space

compile:
gcc make_intensity.c -O3 -lfftw3 -lm -o make_intensity

usage:
./make_intensity filtered-DensityMap.dat qmax

makes:
intensity.dat

*/

#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>

#define PI ((double) 3.14159265359)

int N0, N1, N2;
int qmax ;
double *HKL;

void intens_cal();

int main(int argc, char *argv[]){
	
	int i, j, k, d0, d1, d2 ;

	// Read in electron density map to HKL
	if (argc > 1){
		FILE *fin = fopen(argv[1], "r") ;
		fscanf(fin, "%d %d %d", &d0, &d1, &d2) ;

		qmax = atoi(argv[2]) ;

		// Dimensions of the HKL matrix
		N0 = 2*qmax+1 ;
		N1 = 2*qmax+1 ;
		N2 = 2*qmax+1 ;

		HKL = malloc(sizeof(double) *N0*N1*N2) ;

		for (i = 0; i < N0; i++){
			for (j = 0; j < N1; j++){
				for (k = 0; k < N2; k++){
					if (abs(i-(N0-1)/2) < (d0+1)/2 && abs(j-(N1-1)/2) < (d1+1)/2 && abs(k-(N2-1)/2) < (d2+1)/2)
						fscanf(fin, "%lf", &HKL[k + j*N2 + i*N1*N2]) ;
					else
						HKL[k + j*N2 + i*N1*N2] = 0 ;
				}
			}
		}
		fclose(fin) ;
	}
	else{
		fprintf(stderr, "Give me atom coordinates and qmax!!!\n") ;
		exit(0) ;
	}
	
	intens_cal();

	FILE *outFp = fopen("intensity.dat", "w") ;
	for (i = 0; i < N0; i++){
		for (j = 0; j < N1; j++){
			for (k = 0; k < N2; k++){
				fprintf(outFp, "%lf\n", HKL[k + j*N2 + i*N1*N2]) ;
			}
		}
	}

	fclose(outFp) ;
	free(HKL) ;

	return 0 ;
}


void intens_cal(){
	
	int i, j, k;

	// create an FFTW plan: in (real pace); out (k space)
	fftw_complex *in,*out;
	fftw_plan forward_p,inverse_p;
	in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N0*N1*N2);
	out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N0*N1*N2);
	forward_p = fftw_plan_dft_3d(N0, N1, N2, in, out, FFTW_FORWARD, FFTW_MEASURE);
	inverse_p = fftw_plan_dft_3d(N0, N1, N2, out, in, FFTW_BACKWARD, FFTW_MEASURE);
	
	for (i = 0; i < N0; i++){
		for (j = 0; j < N1; j++){
			for (k = 0; k < N2; k++){
				in[k + j*N2 + i*N1*N2][0] = HKL[k + j*N2 + i*N1*N2];
				in[k + j*N2 + i*N1*N2][1] = 0;
			}
		}
	}
	
	// fft
	fftw_execute(forward_p);
	
	// fftshift
	int id0, id1, id2;
	for (i = 0; i < N0; i++){
		for (j = 0; j < N1; j++){
			for (k = 0; k < N2; k++){
				if (i < (N0+1)/2)
					id0 = i + (N0-1)/2;
				else
					id0 = i - (N0+1)/2;
				if (j < (N1+1)/2)
					id1 = j + (N1-1)/2;
				else
					id1 = j - (N1+1)/2;
				if (k < (N2+1)/2)
					id2 = k + (N2-1)/2;
				else
					id2 = k - (N2+1)/2;
				HKL[id2 + id1*N2 + id0*N1*N2] = pow(out[k + j*N2 + i*N1*N2][0],2) + pow(out[k + j*N2 + i*N1*N2][1],2);
			}
		}
	}
	
	fftw_destroy_plan(forward_p);
	fftw_destroy_plan(inverse_p);
}
