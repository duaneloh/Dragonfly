#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

int main(int argc, char *argv[]) {
	FILE *fp ;
	int i[3], x, y, z ;
	int size = 0, center, spotnum[3] ;
	double t[3], a[3] ;
	double val, r, dist ;
	double *object ;
	int vox[3] ;
	
	if (argc < 6) {
		fprintf(stderr, "Format: %s <recon_fname> <ax> <ay> <az> <radius>\n", argv[0]) ;
		return 1 ;
	}
	a[0] = atof(argv[2]) ;
	a[1] = atof(argv[3]) ;
	a[2] = atof(argv[4]) ;
	r = atof(argv[5]) ;
	
	char line[999], *token, config_fname[999] = "config.ini" ;
	
	fp = fopen(config_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", config_fname) ;
		return 1 ;
	}
	while (fgets(line, 999, fp) != NULL) {
		token = strtok(line, " =") ;
		if (token[0] == '#' || token[0] == '\n' || token[0] == '[')
			continue ;
		
		if (strcmp(token, "size") == 0)
			size = atoi(strtok(NULL, " =\n")) ;
	}
	fclose(fp) ;
	center = size / 2 ;
	
	fp = fopen(argv[1], "rb") ;
	object = malloc(size * size * size * sizeof(double)) ;
	fread(object, sizeof(double), size*size*size, fp) ;
	fclose(fp) ;
	
	for (x = 0 ; x < 3 ; ++x)
		spotnum[x] = (int) 2 * floor(size / a[x] / 2.) + 1 ;
	
	char filename[500] ;
	sprintf(filename, "%s.dat", remove_ext(extract_fname(argv[1]))) ;
	fp = fopen(filename, "w") ;
	
	for (i[0] = -spotnum[0] ; i[0] < spotnum[0] ; ++i[0])
	for (i[1] = -spotnum[1] ; i[1] < spotnum[1] ; ++i[1])
	for (i[2] = -spotnum[2] ; i[2] < spotnum[2] ; ++i[2]) {
		dist = 0. ;
		for (x = 0 ; x < 3 ; ++x) {
			// Reciprocal space coordinates
			t[x] = a[x] * i[x] ;
			
			// Calculate distance from origin
			dist += t[x] * t[x] ;
			
			// Center moved to (center,center,center) to make all coordinates positive.
			t[x] += center ;
		}
		
		// Exclude if out of bounds
		if (t[0] - r < 0. || t[0] + r >= size || 
			t[1] - r < 0. || t[1] + r >= size || 
			t[2] - r < 0. || t[2] + r >= size)
				continue ;
		
		val = 0. ;
		vox[0] = (int) (t[0] + 0.5) ;
		vox[1] = (int) (t[1] + 0.5) ;
		vox[2] = (int) (t[2] + 0.5) ;
		// Convolve with Gaussian spot volume
		for (x = (int) t[0] - r ; x < (int) t[0] + r ; ++x)
		for (y = (int) t[1] - r ; y < (int) t[1] + r ; ++y)
		for (z = (int) t[2] - r ; z < (int) t[2] + r ; ++z)
		if ((x - vox[0])*(x - vox[0]) + (y - vox[1])*(y - vox[1]) + (z - vox[2])*(z - vox[2]) < r*r) {
			val += object[x*size*size + y*size + z] ;
			object[x*size*size + y*size + z] = 0. ;
		}
		
		if (val > 0.)
			fprintf(fp, "%d\t%d\t%d\t%.10e\n", i[0], i[1], i[2], val) ;
	}
	fclose(fp) ;
	fprintf(stderr, "Generated %s with %d x %d x %d spots\n", 
		filename, spotnum[0], spotnum[1], spotnum[2]) ;
	
	sprintf(filename, "%s_red.bin", remove_ext(argv[1])) ;
	FILE *fp_red = fopen(filename, "wb") ;
	fwrite(object, sizeof(double), size*size*size, fp_red) ;
	fclose(fp_red) ;
	
	free(object) ;
	
	return 0 ;
}
