#include "iterate.h"

void calc_frame_counts(struct iterate *self) {
	long t, d, detn, dset = 0 ;
	struct dataset *curr = self->dset ;
	struct detector *cdet ;
	int *num_data = calloc(self->tot_num_data, sizeof(int)) ;
	
	if (self->fcounts == NULL)
		self->fcounts = calloc(self->tot_num_data, sizeof(int)) ;
	if (self->mean_count == NULL)
		self->mean_count = calloc(self->num_det, sizeof(double)) ;
	
	while (curr != NULL) {
		cdet = curr->det ;
		detn = self->det_mapping[dset] ;
		
		if (curr->ftype == SPARSE) {
			for (d = 0 ; d < curr->num_data ; ++d) {
				for (t = 0 ; t < curr->ones[d] ; ++t)
				if (cdet->raw_mask[curr->place_ones[curr->ones_accum[d] + t]] < 1)
					self->fcounts[curr->num_offset + d]++ ;
				
				for (t = 0 ; t < curr->multi[d] ; ++t)
				if (cdet->raw_mask[curr->place_multi[curr->multi_accum[d] + t]] < 1)
					self->fcounts[curr->num_offset + d] += curr->count_multi[curr->multi_accum[d] + t] ;
				
				self->mean_count[detn] += self->fcounts[curr->num_offset + d] ;
				num_data[detn]++ ;
			}
		}
		else if (curr->ftype == DENSE_INT) {
			for (d = 0 ; d < curr->num_data ; ++d) {
				for (t = 0 ; t < curr->num_pix ; ++t)
					self->fcounts[curr->num_offset + d] += curr->int_frames[d*curr->num_pix + t] ;
				
				self->mean_count[detn] += self->fcounts[curr->num_offset + d] ;
				num_data[detn]++ ;
			}
		}
		else if (curr->ftype == DENSE_DOUBLE) {
			for (d = 0 ; d < curr->num_data ; ++d) {
				for (t = 0 ; t < curr->num_pix ; ++t)
					self->fcounts[curr->num_offset + d] += curr->frames[d*curr->num_pix + t] ;
				
				self->mean_count[detn] += self->fcounts[curr->num_offset + d] ;
				num_data[detn]++ ;
			}
		}
		
		dset++ ;
		curr = curr->next ;
	}
	
	for (d = 0 ; d < self->num_det ; ++d)
		self->mean_count[d] /= num_data[d] ;
	free(num_data) ;
}

void calc_beta(double start, struct iterate *self) {
	int d ;

	self->beta_start = malloc(self->tot_num_data * sizeof(double)) ;
	self->beta = malloc(self->tot_num_data * sizeof(double)) ;
	
	if (!self->par->need_scaling && start < 0)
		start = exp(-6.5 * pow(self->mean_count[0] * 1.e-5, 0.15)) ; // Empirical
	
	if (start > 0) {
		for (d = 0 ; d < self->tot_num_data ; ++d)
			self->beta_start[d] = start ;
	}
	else {
		for (d = 0 ; d < self->tot_num_data ; ++d)
			self->beta_start[d] = exp(-6.5 * pow(self->fcounts[d] * 1.e-5, 0.15)) ; // Empirical
	}
}

