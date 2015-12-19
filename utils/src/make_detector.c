#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

int main(int argc, char *argv[]) {
	int i, j, detcen, detsize ;
	double x, y, qx, qy, qz ;
	double stoprad, norm, solid_angle, lambda, detd, qscale ;
	char line[999], *token ;
	char det_fname[999], mask_fname[999] ;
	uint8_t *mask ;
	
	qscale = 0. ;
	detd = 0. ;
	lambda = 0. ;
	detsize = 0 ;
	stoprad = 0 ;
	
	// Read config file
	FILE *fp = fopen("config.ini", "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file config.ini not found.\n") ;
		return 1 ;
	}
	while (fgets(line, 999, fp) != NULL) {
		token = strtok(line, " =") ;
		if (token[0] == '#' || token[0] == '\n' || token[0] == '[')
			continue ;
		
		if (strcmp(token, "detsize") == 0)
			detsize = atoi(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "lambda") == 0)
			lambda = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "detd") == 0)
			detd = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "qscale") == 0)
			qscale = atof(strtok(NULL, " =\n")) ;
		else if (strcmp(token, "detector") == 0)
			strcpy(det_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "mask") == 0)
			strcpy(mask_fname, strtok(NULL, " =\n")) ;
		else if (strcmp(token, "stoprad") == 0)
			stoprad = atof(strtok(NULL, " =\n")) ;
	}
	fclose(fp) ;
	fprintf(stderr, "Parsed config.ini\n") ;
	
	if (detd == 0.) {
		fprintf(stderr, "Need detd (detectorr distance in pixel units)\n") ;
		return 1 ;
	}
	if (lambda == 0.) {
		fprintf(stderr, "Need lambda (beam wavelength in A)\n") ;
		return 1 ;
	}
	if (qscale == 0.) {
		fprintf(stderr, "Need qscale (voxel size in A^-1)\n") ;
		return 1 ;
	}
	if (detsize == 0) {
		fprintf(stderr, "Need detsize (number of pixels along a detector edge)\n") ;
		return 1 ;
	}
	if (stoprad == 0) {
		fprintf(stderr, "Need stoprad (radius of beam stop)\n") ;
		return 1 ;
	}
	detcen = detsize / 2 ;
	qscale = 1. / lambda / qscale ; // Number of voxels needed for q = 1/lambda
	mask = calloc(detsize*detsize, sizeof(double)) ;
	
	// Write detector file
	FILE *outFp = fopen(det_fname, "w") ;
	fprintf(outFp, "%d %f\n", detsize*detsize, detd);
	
	for (i = 0 ; i < detsize ; i++)
	for (j = 0 ; j < detsize ; j++) {
		x = i - detcen ;
		y = j - detcen ;
		
		norm = sqrt(x*x + y*y + detd*detd) ;
		qx = x * qscale / norm ;
		qy = y * qscale / norm ;
		qz = qscale * (detd / norm - 1.) ;
		
		if (x*x + y*y < stoprad*stoprad)
			mask[i*detsize + j] = 2 ;
		if (x*x + y*y > detcen*detcen)
			mask[i*detsize + j] = 1 ;
		
		// Solid angle and polarization correction
		solid_angle = pow(1. + qz / qscale, 3.) * (1. - pow(qy / qscale, 2.)) ;
//		solid_angle = 1*1*detd/pow(sqrt(x*x + y*y + detd*detd), 3) ;
		
		fprintf(outFp, "%21.15e ", qx);
		fprintf(outFp, "%21.15e ", qy);
		fprintf(outFp, "%21.15e ", qz);
		fprintf(outFp, "%21.15e\n", solid_angle) ;
		
		if (j == 0)
			fprintf(stderr, "\rFinished pixel %d/%d", i*detsize, detsize*detsize) ;
	}
	fprintf(stderr, "\rFinished pixel %d/%d\n", detsize*detsize, detsize*detsize) ;
	
	fclose(outFp) ;
	
	// Write mask file
	fp = fopen(mask_fname, "wb") ;
	fwrite(mask, sizeof(uint8_t), detsize*detsize, fp) ;
	fclose(fp) ;
	
	free(mask) ;
	
	return 0 ;
}
