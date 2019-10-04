#include "emcfile.h"

static int parse_binarydataset(char *fname, struct dataset *self) {
	int d ;
	struct detector *det = self->det ;
	
	FILE *fp = fopen(fname, "rb") ;
	fread(&(self->num_data), sizeof(int), 1, fp) ;
	fread(&(self->num_pix), sizeof(int) , 1, fp) ;
	if (self->num_pix != det->num_pix)
		fprintf(stderr, "WARNING! The detector file and photons file %s do not "
		                "have the same number of pixels\n", self->fname) ;
	fread(&(self->ftype), sizeof(int), 1, fp) ;
	fseek(fp, 1024, SEEK_SET) ;
	if (self->ftype == SPARSE) {
		self->ones = malloc(self->num_data * sizeof(int)) ;
		self->multi = malloc(self->num_data * sizeof(int)) ;
		fread(self->ones, sizeof(int), self->num_data, fp) ;
		fread(self->multi, sizeof(int), self->num_data, fp) ;
		
		self->ones_accum = malloc(self->num_data * sizeof(long)) ;
		self->multi_accum = malloc(self->num_data * sizeof(long)) ;
		self->ones_accum[0] = 0 ;
		self->multi_accum[0] = 0 ;
		for (d = 1 ; d < self->num_data ; ++d) {
			self->ones_accum[d] = self->ones_accum[d-1] + self->ones[d-1] ;
			self->multi_accum[d] = self->multi_accum[d-1] + self->multi[d-1] ;
		}
		self->ones_total = self->ones_accum[self->num_data-1] + self->ones[self->num_data-1] ;
		self->multi_total = self->multi_accum[self->num_data-1] + self->multi[self->num_data-1] ;
		
		self->place_ones = malloc(self->ones_total * sizeof(int)) ;
		self->place_multi = malloc(self->multi_total * sizeof(int)) ;
		self->count_multi = malloc(self->multi_total * sizeof(int)) ;
		fread(self->place_ones, sizeof(int), self->ones_total, fp) ;
		fread(self->place_multi, sizeof(int), self->multi_total, fp) ;
		fread(self->count_multi, sizeof(int), self->multi_total, fp) ;
	}
	else if (self->ftype == DENSE_INT) {
		fprintf(stderr, "%s is a dense integer emc file\n", self->fname) ;
		self->int_frames = malloc(self->num_pix * self->num_data * sizeof(int)) ;
		fread(self->int_frames, sizeof(int), self->num_pix * self->num_data, fp) ;
	}
	else if (self->ftype == DENSE_DOUBLE) {
		fprintf(stderr, "%s is a dense double precision emc file\n", self->fname) ;
		self->frames = malloc(self->num_pix * self->num_data * sizeof(double)) ;
		fread(self->frames, sizeof(double), self->num_pix * self->num_data, fp) ;
	}
	else {
		fprintf(stderr, "Unknown dataset type %d\n", self->ftype) ;
		fclose(fp) ;
		return 1 ;
	}
	fclose(fp) ;

	return 0 ;
}

static int parse_h5dataset(char *fname, struct dataset *self) {
	int d ;
	struct detector *det = self->det ;
	hid_t file, dset, dspace, dtype ;
	hsize_t bufsize ;
	hvl_t *buffer ;
	
	file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT) ;
	
	dset = H5Dopen(file, "num_pix", H5P_DEFAULT) ;
	H5Dread(dset, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(self->num_pix)) ;
	H5Dclose(dset) ;
	if (self->num_pix != det->num_pix)
		fprintf(stderr, "WARNING! The detector file and photons file %s do not "
		                "have the same number of pixels\n", self->fname) ;
	
	dset = H5Dopen(file, "place_ones", H5P_DEFAULT) ;
	dtype = H5Tvlen_create(H5T_STD_I32LE) ;
	dspace = H5Dget_space(dset) ;
	H5Sget_simple_extent_dims(dspace, &bufsize, NULL) ;
	self->num_data = bufsize ;
	buffer = malloc(self->num_data * sizeof(hvl_t)) ;
	H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) ;
	self->ones = malloc(self->num_data * sizeof(int)) ;
	self->ones_accum = malloc(self->num_data * sizeof(long)) ;
	H5Dvlen_get_buf_size(dset, dtype, dspace, &bufsize) ;
	self->ones_total = bufsize / sizeof(int) ;
	self->place_ones = malloc(self->ones_total * sizeof(int)) ;
	self->ones_accum[0] = 0 ;
	self->ones[0] = buffer[0].len ;
	memcpy(self->place_ones, buffer[0].p, sizeof(int)*buffer[0].len) ;
	free(buffer[0].p) ;
	for (d = 1 ; d < self->num_data ; ++d) {
		self->ones[d] = buffer[d].len ;
		self->ones_accum[d] = self->ones_accum[d-1] + self->ones[d-1] ;
		memcpy(&self->place_ones[self->ones_accum[d]], buffer[d].p, sizeof(int)*buffer[d].len) ;
		free(buffer[d].p) ;
	}
	if (self->ones_total != self->ones_accum[self->num_data-1] + self->ones[self->num_data-1])
		fprintf(stderr, "WARNING: ones_total mismatch in %s\n", self->fname) ;
	H5Dclose(dset) ;
	
	dset = H5Dopen(file, "place_multi", H5P_DEFAULT) ;
	H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) ;
	self->multi = malloc(self->num_data * sizeof(int)) ;
	self->multi_accum = malloc(self->num_data * sizeof(long)) ;
	H5Dvlen_get_buf_size(dset, dtype, dspace, &bufsize) ;
	self->multi_total = bufsize / sizeof(int) ;
	self->place_multi = malloc(self->multi_total * sizeof(int)) ;
	self->multi_accum[0] = 0 ;
	self->multi[0] = buffer[0].len ;
	memcpy(self->place_multi, buffer[0].p, sizeof(int)*buffer[0].len) ;
	free(buffer[0].p) ;
	for (d = 1 ; d < self->num_data ; ++d) {
		self->multi[d] = buffer[d].len ;
		self->multi_accum[d] = self->multi_accum[d-1] + self->multi[d-1] ;
		memcpy(&self->place_multi[self->multi_accum[d]], buffer[d].p, sizeof(int)*buffer[d].len) ;
		free(buffer[d].p) ;
	}
	if (self->multi_total != self->multi_accum[self->num_data-1] + self->multi[self->num_data-1])
		fprintf(stderr, "WARNING: multi_total mismatch in %s\n", self->fname) ;
	H5Dclose(dset) ;
	
	dset = H5Dopen(file, "count_multi", H5P_DEFAULT) ;
	H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) ;
	self->count_multi = malloc(self->multi_total * sizeof(int)) ;
	for (d = 0 ; d < self->num_data ; ++d) {
		memcpy(&self->count_multi[self->multi_accum[d]], buffer[d].p, sizeof(int)*buffer[d].len) ;
		free(buffer[d].p) ;
	}
	H5Dclose(dset) ;
	free(buffer) ;
	
	H5Sclose(dspace) ;
	H5Tclose(dtype) ;
	H5Fclose(file) ;
	
	return 0 ;
}

int parse_dataset(char *fname, struct detector *det, struct dataset *self) {
	int err ;
	long d, t ;
	char line[1024], hdfheader[8] = {137, 'H', 'D', 'F', '\r', '\n', 26, '\n'} ;
	
	self->det = det ;
	self->ones_total = 0, self->multi_total = 0 ;
	strcpy(self->fname, fname) ;
	
	FILE *fp = fopen(fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "data_fname %s not found. Exiting.\n", fname) ;
		return 1 ;
	}
	fread(line, 8, sizeof(char), fp) ;
	fclose(fp) ;
	
	if (strncmp(line, hdfheader, 8) == 0) {
		fprintf(stderr, "Parsing HDF5 dataset %s\n", fname) ;
		self->ftype = SPARSE ;
		err = parse_h5dataset(fname, self) ;
	}
	else {
		err = parse_binarydataset(fname, self) ;
	}
	if (err)
		return err ;
	
	// Calculate mean count in the presence of mask
	self->mean_count = 0. ;
	for (d = 0 ; d < self->num_data ; ++d) {
		if (self->ftype == SPARSE) {
			for (t = 0 ; t < self->ones[d] ; ++t)
			if (det->raw_mask[self->place_ones[self->ones_accum[d] + t]] < 1)
				self->mean_count += 1. ;
			
			for (t = 0 ; t < self->multi[d] ; ++t)
			if (det->raw_mask[self->place_multi[self->multi_accum[d] + t]] < 1)
				self->mean_count += self->count_multi[self->multi_accum[d] + t] ;
		}
		else if (self->ftype == DENSE_INT) {
			for (t = 0 ; t < self->num_pix ; ++t)
			if (det->raw_mask[t] < 1)
				self->mean_count += self->int_frames[d*self->num_pix + t] ;
		}
		else if (self->ftype == DENSE_DOUBLE) {
			for (t = 0 ; t < self->num_pix ; ++t)
			if (det->raw_mask[t] < 1)
				self->mean_count += self->frames[d*self->num_pix + t] ;
		}
	}
	
	self->mean_count /= self->num_data ;
	
	return err ;
}

