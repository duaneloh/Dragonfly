#ifndef ITERATE_H
#define ITERATE_H

#include "detector.h"
#include "emcfile.h"
#include "quaternion.h"
#include "model.h"

struct iterate {
	struct detector *det ;
	struct model *mod ;
	struct dataset *dset ;
	struct quaternion *quat ;

	// Parameters for each frame
	int tot_num_data, num_blacklist ;
	int *fcounts ;
	double *scale, *bgscale ;
	uint8_t *blacklist ;

	// For refinement
	int *quat_mapping, **rel_quat, *num_rel_quat ;
	double **rel_prob ;

	// Parameters for each detector
	int num_det ; // Number of unique detectors
	int *det_mapping ; // Mapping to unique list
	double *rescale, *mean_count ;

	// Aggregate metrics
	double likelihood, mutual_info, rms_change ;

    // Params
    int update_scale ;
} ;

void calc_frame_counts(struct iterate*) ;

#endif // ITERATE_H
