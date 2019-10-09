#include "iterate.h"

void calc_frame_counts(struct iterate *self) {
	long t, d ;
	struct dataset *curr = self->dset ;
	struct detector *cdet ;
	
	if (self->fcounts == NULL)
		self->fcounts = calloc(self->tot_num_data, sizeof(int)) ;

	while (curr != NULL) {
		cdet = curr->det ;
		if (curr->ftype == SPARSE) {
			for (d = 0 ; d < curr->num_data ; ++d) {
				for (t = 0 ; t < curr->ones[d] ; ++t)
				if (cdet->raw_mask[curr->place_ones[curr->ones_accum[d] + t]] < 1)
					self->fcounts[curr->num_offset + d]++ ;
				
				for (t = 0 ; t < curr->multi[d] ; ++t)
				if (cdet->raw_mask[curr->place_multi[curr->multi_accum[d] + t]] < 1)
					self->fcounts[curr->num_offset + d] += curr->count_multi[curr->multi_accum[d] + t] ;
			}
		}
		else if (curr->ftype == DENSE_INT) {
			for (d = 0 ; d < curr->num_data ; ++d)
			for (t = 0 ; t < curr->num_pix ; ++t)
				self->fcounts[curr->num_offset + d] += curr->int_frames[d*curr->num_pix + t] ;
		}
		else if (curr->ftype == DENSE_DOUBLE) {
			for (d = 0 ; d < curr->num_data ; ++d)
			for (t = 0 ; t < curr->num_pix ; ++t)
				self->fcounts[curr->num_offset + d] += curr->frames[d*curr->num_pix + t] ;
		}
		
		curr = curr->next ;
	}
}

