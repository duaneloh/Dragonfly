#include "maximize.h"

#define PROB_MIN 1.e-6
#define PDIFF_THRESH 14.
#define MAX_EXP_START -1.e100

static struct timeval tm1, tm2 ;

// Start Maximize
static void allocate_memory(struct max_data*) ;
static void calculate_rescale(struct max_data*) ;
// -- Start OpenMP
// -- for r in {0..num_rot}
	static void calculate_prob(int, struct max_data*, struct max_data*) ;
static void normalize_prob(struct max_data*, struct max_data*) ;
// -- for r in {0..num_rot}
	static void update_tomogram(int, struct max_data*, struct max_data*) ;
	static void merge_tomogram(int, struct max_data*) ;
static void combine_information_omp(struct max_data*, struct max_data*) ;
// -- End OpenMP
static double combine_information_mpi(struct max_data*) ;
static void update_scale(struct max_data*) ;
// End Maximize

// Other functions
static int resparsify(double*, int*, int, double) ;
static double calc_psum_r(int, struct max_data*, struct max_data*) ;
static void print_max_time(char*, char*, int) ;

void (*slice_gen)(double*, int, double*, struct detector*, struct model*) ;
void (*slice_merge)(double*, int, double*, struct detector*, struct model*) ;

double maximize(struct max_data *common_data) {
	double avg_likelihood ;
	struct iterate *iter = common_data->iter ;
	if (iter == NULL) {
		fprintf(stderr, "No iterate in max_data!\n") ;
		return -1. ;
	}
	
	struct quaternion *quat = iter->quat ;
	struct params *param = iter->par ;

	gettimeofday(&tm1, NULL) ;
	common_data->within_openmp = 0 ;
	
	allocate_memory(common_data) ;
	calculate_rescale(common_data) ;

	#pragma omp parallel default(shared)
	{
		int r ;
		struct max_data *priv_data = malloc(sizeof(struct max_data)) ;
		
		priv_data->within_openmp = 1 ;
		priv_data->iter = iter ;
		allocate_memory(priv_data) ;
		
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < quat->num_rot_p ; ++r)
			calculate_prob(r, priv_data, common_data) ;
		
		normalize_prob(priv_data, common_data) ;
		
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < quat->num_rot_p ; ++r) {
			update_tomogram(r, priv_data, common_data) ;
			merge_tomogram(r, priv_data) ;
		}
		
		combine_information_omp(priv_data, common_data) ;
		
		free_max_data(priv_data) ;
	}

	avg_likelihood = combine_information_mpi(common_data) ;
	if (param->need_scaling && param->update_scale)
		update_scale(common_data) ;
	
	return avg_likelihood ;
}

void allocate_memory(struct max_data *data) {
	int detn, d ;
	struct iterate *iter = data->iter ;
	struct model *mod = iter->mod ;
	struct detector *det = iter->det ;
	struct quaternion *quat = iter->quat ;
	struct params *param = iter->par ;
	
	// Both private and common
	data->rmax = calloc(iter->tot_num_data, sizeof(int)) ;
	data->info = calloc(iter->tot_num_data, sizeof(double)) ;
	data->likelihood = calloc(iter->tot_num_data, sizeof(double)) ;
	data->max_exp_p = malloc(iter->tot_num_data * sizeof(double)) ;
	for (d = 0 ; d < iter->tot_num_data ; ++d)
		data->max_exp_p[d] = MAX_EXP_START ;
	if (mod->num_modes > 1)
		data->quat_norm = calloc(mod->num_modes * iter->tot_num_data, sizeof(double)) ;
	
	data->prob = malloc(iter->tot_num_data * sizeof(double*)) ;
	data->place_prob = malloc(iter->tot_num_data * sizeof(int*)) ;
	for (d = 0 ; d < iter->tot_num_data ; ++d) {
		data->prob[d] = NULL ;
		data->place_prob[d] = NULL ;
	}
	data->num_prob = calloc(iter->tot_num_data, sizeof(int)) ;
	if (param->need_scaling && param->update_scale)
		data->psum_d = calloc(iter->tot_num_data, sizeof(double)) ;
		
	if (!data->within_openmp) { // common_data
		data->u = malloc(iter->num_det * sizeof(double*)) ;
		for (detn = 0 ; detn < iter->num_det ; ++detn)
			data->u[detn] = calloc(quat->num_rot_p, sizeof(double)) ;
		data->max_exp = calloc(iter->tot_num_data, sizeof(double)) ;
		data->p_norm = calloc(iter->tot_num_data, sizeof(double)) ;
		data->offset_prob = calloc(iter->tot_num_data * omp_get_max_threads(), sizeof(int)) ;
		
		memset(mod->model2, 0, mod->num_modes*mod->vol*sizeof(double)) ;
		memset(mod->inter_weight, 0, mod->num_modes*mod->vol*sizeof(double)) ;
		print_max_time("alloc", "", param->verbosity > 1 && param->rank == 0) ;
	}
	else { // priv_data
		data->all_views = malloc(iter->num_det * sizeof(double*)) ;
		for (d = 0 ; d < iter->num_det ; ++d)
			data->all_views[d] = malloc(det[d].num_pix * sizeof(double)) ;
		
		data->model = calloc(mod->num_modes*mod->vol, sizeof(double)) ;
		data->weight = calloc(mod->num_modes*mod->vol, sizeof(double)) ;
		
		data->psum_r = calloc(iter->num_det, sizeof(double)) ;
		
		for (d = 0 ; d < iter->tot_num_data ; ++d) {
			data->prob[d] = malloc(4 * sizeof(double)) ;
			data->place_prob[d] = malloc(4 * sizeof(int)) ;
		}
	}
}

void calculate_rescale(struct max_data *data) {
	int detn ;
	struct iterate *iter = data->iter ;
	struct model *mod = iter->mod ;
	struct detector *det = iter->det ;
	struct quaternion *quat = iter->quat ;
	struct params *param = iter->par ;
	double *total = calloc(iter->num_det, sizeof(double)) ;
	char res_string[1024] = {'\0'}  ;
	
	// Calculate rescale factor by calculating mean model value over detector
	#pragma omp parallel default(shared)
	{
		int r, t, detn, mode, rotind ;
		double *priv_total = calloc(iter->num_det, sizeof(double)) ;
		double **views = malloc(iter->num_det * sizeof(double*)) ;
		for (detn = 0 ; detn < iter->num_det ; ++detn)
			views[detn] = malloc(det[detn].num_pix * sizeof(double)) ;
		
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < quat->num_rot_p ; ++r) {
			rotind = (r*param->num_proc + param->rank) / param->num_modes ;
			mode = (r*param->num_proc + param->rank) % param->num_modes ;
			if (rotind >= quat->num_rot) {
				mode = r*param->num_proc + param->rank - param->num_modes * (quat->num_rot - 1) ;
				rotind = 0 ;
			}
			//fprintf(stderr, "%d: %.3d - %.2d %.2d\n", omp_get_thread_num(), r, rotind, mode) ;
			
			for (detn = 0 ; detn < iter->num_det ; ++detn) {
				// Second argument being 0. tells slice_gen to generate un-rescaled tomograms
				(*slice_gen)(&quat->quats[rotind*5], mode, views[detn], &det[detn], mod) ;
				
				for (t = 0 ; t < det[detn].num_pix ; ++t)
				if (det[detn].raw_mask[t] < 1)
					data->u[detn][r] += views[detn][t] ;
				
				priv_total[detn] += quat->quats[rotind*5 + 4] * data->u[detn][r] ;
			}
		}
		
		#pragma omp critical(total)
		{
			for (detn = 0 ; detn < iter->num_det ; ++detn)
				total[detn] += priv_total[detn] ;
		}
		for (detn = 0 ; detn < iter->num_det ; ++detn)
			free(views[detn]) ;
		free(views) ;
		free(priv_total) ;
	}
	
	MPI_Allreduce(MPI_IN_PLACE, total, iter->num_det, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	sprintf(res_string, "(=") ;
	for (detn = 0 ; detn < iter->num_det ; ++detn) {
		iter->rescale[detn] = iter->mean_count[detn] / total[detn] * mod->num_modes ;
		sprintf(res_string + strlen(res_string), " %.6e", iter->rescale[detn]) ;
	}
	sprintf(res_string + strlen(res_string), ")") ;
	print_max_time("rescale", res_string, param->verbosity > 1 && param->rank == 0) ;
}

void calculate_prob(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, t, d, curr_d, pixel, mode, rotind, detn, old_detn = -1, ind ;
	struct iterate *iter = common->iter ;
	struct dataset *frames = iter->dset ;
	struct model *mod = iter->mod ;
	struct detector *det = iter->det ;
	struct quaternion *quat = iter->quat ;
	struct params *param = iter->par ;
	
	struct dataset *curr = frames ;
	double pval, *view ;
	int *num_prob = priv->num_prob, **place_prob = priv->place_prob ;
	double **prob = priv->prob ;
	
	rotind = (r*param->num_proc + param->rank) / param->num_modes ;
	mode = (r*param->num_proc + param->rank) % param->num_modes ;
	if (rotind >= quat->num_rot) {
		mode = r*param->num_proc + param->rank - param->num_modes * (quat->num_rot - 1) ;
		rotind = 0 ;
	}
	
	// Linked list of data sets from different files
	while (curr != NULL) {
		// Calculate slice for current detector
		detn = iter->det_mapping[dset] ;
		view = priv->all_views[detn] ;
		if (detn != old_detn) {
			(*slice_gen)(&quat->quats[rotind*5], mode, view, &det[detn], mod) ;
			for (t = 0 ; t < det[detn].num_pix ; ++t)
				view[t] = log(view[t]) ;
		}
		old_detn = detn ;
		
		// For each frame in data set
		for (curr_d = 0 ; curr_d < curr->num_data ; ++curr_d) {
			// Calculate frame number in full list
			d = curr->num_offset + curr_d ;
			
			// check if frame is blacklisted
			if (iter->blacklist[d])
				continue ;
			
			// For refinement, check if frame should be processed
			if (param->refine) {
				ind = -1 ;
				for (t = 0 ; t < iter->num_rel_quat[d] ; ++t)
				if (iter->quat_mapping[r*param->num_proc+param->rank] == iter->rel_quat[d][t]) {
					ind = t ;
					break ;
				}
				if (ind == -1)
					continue ;
			}
			
			if (curr->ftype < 2) {
				// need_scaling is for if we want to assume variable incident intensity
				if (param->need_scaling && (param->iteration > 1 || param->known_scale))
					pval = log(quat->quats[rotind*5 + 4]) - common->u[detn][r] * iter->scale[d] ;
				else
					pval = log(quat->quats[rotind*5 + 4]) - common->u[detn][r] * iter->rescale[detn] ;
			}
			else {
				pval = 0. ;
			}
			
			if (curr->ftype == SPARSE) {
				// For each pixel with one photon
				for (t = 0 ; t < curr->ones[curr_d] ; ++t) {
					pixel = curr->place_ones[curr->ones_accum[curr_d] + t] ;
					if (det[detn].raw_mask[pixel] < 1)
						pval += view[pixel] ;
				}
				
				// For each pixel with count_multi photons
				for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
					pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
					if (det[detn].raw_mask[pixel] < 1)
						pval += curr->count_multi[curr->multi_accum[curr_d] + t] * view[pixel] ;
				}
			}
			else if (curr->ftype == DENSE_INT) {
				for (t = 0 ; t < det[detn].num_pix ; ++t)
				if (det[detn].raw_mask[t] < 1)
					pval += curr->int_frames[curr_d*curr->num_pix + t] * view[t] ;
			}
			else if (curr->ftype == DENSE_DOUBLE) { // Gaussian EMC for double precision data without scaling
				for (t = 0 ; t < det[detn].num_pix ; ++t)
				if (det[detn].raw_mask[t] < 1)
					pval -= pow(curr->frames[curr_d*curr->num_pix + t] - view[t]*iter->rescale[detn], 2.) ;
			}
			
			// Only save value in prob array if it is significant
			if (pval + PDIFF_THRESH/iter->beta[d] > priv->max_exp_p[d]) {
				prob[d][num_prob[d]] = pval ;
				place_prob[d][num_prob[d]] = r*param->num_proc + param->rank ;
				num_prob[d]++ ;
				
				// If num_prob is a power of two, expand array
				if (num_prob[d] >= 4 && (num_prob[d] & (num_prob[d] - 1)) == 0) {
					prob[d] = realloc(prob[d], num_prob[d] * 2 * sizeof(double)) ;
					place_prob[d] = realloc(place_prob[d], num_prob[d] * 2 * sizeof(int)) ;
				}
			}
			
			// Note maximum log-likelihood for each frame among 'r's tested by this MPI rank and OMP rank
			// Recalculate sparse array with new maximum
			if (pval > priv->max_exp_p[d]) {
				priv->max_exp_p[d] = pval ;
				priv->rmax[d] = r*param->num_proc + param->rank ;
				num_prob[d] = resparsify(prob[d], place_prob[d], num_prob[d], pval - PDIFF_THRESH/iter->beta[d]) ;
			}
		}
		
		curr = curr->next ;
		dset++ ;
	}
	
	if ((param->verbosity > 2) && ((r*param->num_proc + param->rank)%(quat->num_rot * param->num_modes / 10) == 0))
		fprintf(stderr, "\t\tFinished r = %d/%d\n", r*param->num_proc + param->rank, quat->num_rot * param->num_modes + param->nonrot_modes) ;
	print_max_time("prob", "", param->verbosity > 1 && (r == quat->num_rot_p-1) && (param->rank == 0)) ;
}

void normalize_prob(struct max_data *priv, struct max_data *common) {
	int r, d, omp_rank = omp_get_thread_num() ;
	struct iterate *iter = common->iter ;
	struct dataset *frames = iter->dset ;
	struct params *param = iter->par ;
	double *priv_norm = calloc(iter->tot_num_data, sizeof(double)) ;
	
	// Calculate max_log_prob over all OpenMP ranks (and the r for that maximum)
	#pragma omp critical(maxexp)
	{
		for (d = 0 ; d < iter->tot_num_data ; ++d)
		if (priv->max_exp_p[d] > common->max_exp_p[d]) {
			common->max_exp_p[d] = priv->max_exp_p[d] ;
			common->rmax[d] = priv->rmax[d] ;
		}
	}
	#pragma omp barrier
	
	if (omp_rank == 0) {
		MPI_Allreduce(common->max_exp_p, common->max_exp, iter->tot_num_data, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) ;
		
		// Determine 'r' for which log-likelihood is maximum
		for (d = 0 ; d < iter->tot_num_data ; ++d)
		if (common->max_exp[d] != common->max_exp_p[d] || common->max_exp_p[d] == MAX_EXP_START)
			common->rmax[d] = -1 ;
		
		MPI_Allreduce(MPI_IN_PLACE, common->rmax, iter->tot_num_data, MPI_INT, MPI_MAX, MPI_COMM_WORLD) ;
	}
	#pragma omp barrier
	
	for (d = 0 ; d < iter->tot_num_data ; ++d)
	for (r = 0 ; r < priv->num_prob[d] ; ++r) 
	if (frames->ftype < 2)
		priv_norm[d] += exp(iter->beta[d] * (priv->prob[d][r] - common->max_exp[d])) ;
	else
		priv_norm[d] += exp(iter->beta[d] * (priv->prob[d][r] - common->max_exp[d]) / 2. / param->sigmasq) ;
	
	#pragma omp critical(psum)
	{
		for (d = 0 ; d < iter->tot_num_data ; ++d)
			common->p_norm[d] += priv_norm[d] ;
	}
	#pragma omp barrier
	
	if (omp_rank == 0)
		MPI_Allreduce(MPI_IN_PLACE, common->p_norm, iter->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	#pragma omp barrier
	
	free(priv_norm) ;
	print_max_time("norm", "", param->verbosity > 1 && param->rank == 0 && omp_rank == 0) ;
}

void update_tomogram(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, t, d, curr_d, pixel, detn, ind ;
	struct iterate *iter = common->iter ;
	struct dataset *frames = iter->dset ;
	struct detector *det = iter->det ;
	struct params *param = iter->par ;
	double *view ;
	struct dataset *curr = frames ;
	double **prob = priv->prob ;
	int **place_prob = priv->place_prob, *num_prob = priv->num_prob ;
	
	calc_psum_r(r, priv, common) ;
	
	for (detn = 0 ; detn < iter->num_det ; ++detn) 
		memset(priv->all_views[detn], 0, det[detn].num_pix*sizeof(double)) ;
	
	while (curr != NULL) {
		// Calculate slice for current detector
		detn = iter->det_mapping[dset] ;
		view = priv->all_views[detn] ;
		
		for (curr_d = 0 ; curr_d < curr->num_data ; ++curr_d) {
			// Calculate frame number in full list
			d = curr->num_offset + curr_d ;
			
			// check if frame is blacklisted
			if (iter->blacklist[d])
				continue ;
			
			// For refinement, check if frame should be processed
			if (param->refine) {
				ind = -1 ;
				for (t = 0 ; t < iter->num_rel_quat[d] ; ++t)
				if (iter->quat_mapping[r*param->num_proc+param->rank] == iter->rel_quat[d][t]) {
					ind = t ;
					break ;
				}
				if (ind == -1)
					continue ;
			}
			
			// check if current frame has significant probability
			ind = -1 ;
			for (t = 0 ; t < num_prob[d] ; ++t)
			if (r*param->num_proc + param->rank == place_prob[d][t]) {
				ind = t ;
				break ;
			}
			if (ind == -1)
				continue ;
			
			
			// Skip if probability is very low (saves time)
			if (!(prob[d][ind] > PROB_MIN))
				continue ;
			
			if (curr->ftype == 0) {
				// For all pixels with one photon
				for (t = 0 ; t < curr->ones[curr_d] ; ++t) {
					pixel = curr->place_ones[curr->ones_accum[curr_d] + t] ;
					if (det[detn].raw_mask[pixel] < 2)
						view[pixel] += prob[d][ind] ;
				}
				
				// For all pixels with count_multi photons
				for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
					pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
					if (det[detn].raw_mask[pixel] < 2)
						view[pixel] += curr->count_multi[curr->multi_accum[curr_d] + t] * prob[d][ind] ;
				}
			}
			else if (curr->ftype == 1) {
				for (t = 0 ; t < curr->num_pix ; ++t)
					view[t] += curr->int_frames[curr_d*curr->num_pix + t] * prob[d][ind] ;
			}
			else if (curr->ftype == 2) { // Gaussian EMC update without scaling
				for (t = 0 ; t < curr->num_pix ; ++t)
				if (det[detn].raw_mask[t] < 2)
					view[t] += curr->frames[curr_d*curr->num_pix + t] * prob[d][ind] ;
			}
		}
		
		curr = curr->next ;
		dset++ ;
	}

	for (detn = 0 ; detn < iter->num_det ; ++detn) 
	for (t = 0 ; t < det[detn].num_pix ; ++t)
	if (priv->psum_r[detn] > 0.)
		priv->all_views[detn][t] /= priv->psum_r[detn] ;

}

void merge_tomogram(int r, struct max_data *priv) {
	int detn, mode, rotind ;
	struct iterate *iter = priv->iter ;
	struct model *mod = iter->mod ;
	struct detector *det = iter->det ;
	struct quaternion *quat = iter->quat ;
	struct params *param = iter->par ;
	
	rotind = (r*param->num_proc + param->rank) / param->num_modes ;
	mode = (r*param->num_proc + param->rank) % param->num_modes ;
	if (rotind >= quat->num_rot) {
		mode = r*param->num_proc + param->rank - param->num_modes * (quat->num_rot - 1) ;
		rotind = 0 ;
	}
	
	// If no data frame has any probability for this orientation, don't merge
	for (detn = 0 ; detn < iter->num_det ; ++detn)
	if (priv->psum_r[detn] > 0.)
		(*slice_merge)(&quat->quats[rotind*5], mode, priv->all_views[detn], &det[detn], mod) ;
	
	if ((param->verbosity > 2) && ((r*param->num_proc + param->rank)%(quat->num_rot * param->num_modes / 10) == 0))
		fprintf(stderr, "\t\tFinished r = %d/%d\n", r*param->num_proc + param->rank, quat->num_rot * param->num_modes + param->nonrot_modes) ;
}

void combine_information_omp(struct max_data *priv, struct max_data *common) {
	int d, r, omp_rank = omp_get_thread_num() ;
	int nthreads = omp_get_num_threads() ;
	struct iterate *iter = common->iter ;
	struct model *mod = iter->mod ;
	struct params *param = iter->par ;
	long x ;
	
	print_max_time("update", "", param->verbosity > 1 && param->rank == 0 && omp_rank == 0) ;
	 
	#pragma omp critical(model)
	{
		for (x = 0 ; x < mod->num_modes * mod->vol ; ++x) {
			mod->model2[x] += priv->model[x] ;
			mod->inter_weight[x] += priv->weight[x] ;
		}
	}
	
	#pragma omp critical(like_info)
	{
		for (d = 0 ; d < iter->tot_num_data ; ++d) {
			common->likelihood[d] += priv->likelihood[d] ;
			common->info[d] += priv->info[d] ;
		}
	}
	
	// Only calculate common probabilities to save
	if (param->save_prob) {
		// Calculate offsets to combine sparse probabilities for each OpenMP rank
		#pragma omp critical(offset_prob)
		{
			for (d = 0 ; d < iter->tot_num_data ; ++d) {
				common->num_prob[d] += priv->num_prob[d] ;
				for (r = omp_rank + 1 ; r < nthreads ; ++r)
					common->offset_prob[d*nthreads + r] += priv->num_prob[d] ;
			}
		}
		#pragma omp barrier
			
		// Allocate common prob arrays
		#pragma omp for schedule(static,1)
		for (d = 0 ; d < iter->tot_num_data ; ++d) {
			common->prob[d] = malloc(common->num_prob[d] * sizeof(double)) ;
			common->place_prob[d] = malloc(common->num_prob[d] * sizeof(int)) ;
		}
		
		// Populate common->prob array for all d
		for (d = 0 ; d < iter->tot_num_data ; ++d)
		for (r = 0 ; r < priv->num_prob[d] ; ++r) {
			common->prob[d][common->offset_prob[d*nthreads + omp_rank] + r] = priv->prob[d][r] ;
			common->place_prob[d][common->offset_prob[d*nthreads + omp_rank] + r] = priv->place_prob[d][r] ;
		}
		#pragma omp barrier
		
		// Sparsify probs based on PROB_MIN threshold
		#pragma omp for schedule(static,1)
		for (d = 0 ; d < iter->tot_num_data ; ++d)
			common->num_prob[d] = resparsify(common->prob[d], common->place_prob[d], common->num_prob[d], PROB_MIN) ;
	}
	
	if (param->need_scaling && param->update_scale) {
		#pragma omp critical(scale)
		{
			for (d = 0 ; d < iter->tot_num_data ; ++d)
			if (!iter->blacklist[d])
				common->psum_d[d] += priv->psum_d[d] ;
		}
	}
	
	if (mod->num_modes > 1) {
		#pragma omp critical(quat_norm)
		for (d = 0 ; d < iter->tot_num_data * mod->num_modes ; ++d)
			common->quat_norm[d] += priv->quat_norm[d] ;
	}
}

double combine_information_mpi(struct max_data *common) {
	int d ;
	struct iterate *iter = common->iter ;
	struct model *mod = iter->mod ;
	struct params *param = iter->par ;
	double avg_likelihood = 0. ;
	iter->mutual_info = 0. ;
	
	// Combine 3D volumes from all MPI ranks
	if (param->rank) {
		MPI_Reduce(mod->model2, mod->model2, mod->num_modes*mod->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
		MPI_Reduce(mod->inter_weight, mod->inter_weight, mod->num_modes*mod->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
	}
	else {
		MPI_Reduce(MPI_IN_PLACE, mod->model2, mod->num_modes*mod->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
		MPI_Reduce(MPI_IN_PLACE, mod->inter_weight, mod->num_modes*mod->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
	}
	
	// Combine mutual info and likelihood from all MPI ranks
	MPI_Allreduce(MPI_IN_PLACE, common->likelihood, iter->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	MPI_Allreduce(MPI_IN_PLACE, common->info, iter->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	for (d = 0 ; d < iter->tot_num_data ; ++d)
	if (!iter->blacklist[d]) {
		iter->mutual_info += common->info[d] ;
		avg_likelihood += common->likelihood[d] ;
	}
	
	iter->mutual_info /= (iter->tot_num_data - iter->num_blacklist) ;
	avg_likelihood /= (iter->tot_num_data - iter->num_blacklist) ;
	
	// Only calculate common probabilities to save
	if (param->save_prob) {
		int p, q, tot_num_prob ;
		int *num_prob_p = calloc(iter->tot_num_data * param->num_proc, sizeof(int)) ;
		int *displ_prob_p = NULL ;
		for (d = 0 ; d < iter->tot_num_data ; ++d)
			num_prob_p[d*param->num_proc + param->rank] = common->num_prob[d] ;
		if (param->rank) {
			MPI_Reduce(num_prob_p, NULL, iter->tot_num_data*param->num_proc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) ;
			free(num_prob_p) ;
		}
		else {
			MPI_Reduce(MPI_IN_PLACE, num_prob_p, iter->tot_num_data*param->num_proc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) ;
			displ_prob_p = malloc(param->num_proc * sizeof(int)) ;
		}
		for (d = 0 ; d < iter->tot_num_data ; ++d) {
			if (param->rank) {
				MPI_Gatherv(common->place_prob[d], common->num_prob[d], MPI_INT, NULL, 0, 0, MPI_INT, 0, MPI_COMM_WORLD) ;
				MPI_Gatherv(common->prob[d], common->num_prob[d], MPI_DOUBLE, NULL, 0, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
			}
			else {
				tot_num_prob = 0 ;
				memset(displ_prob_p, 0, param->num_proc*sizeof(int)) ;
				for (p = 0 ; p < param->num_proc ; ++p) {
					tot_num_prob += num_prob_p[d*param->num_proc + p] ;
					for (q = 0 ; q < p ; ++q)
						displ_prob_p[p] += num_prob_p[d*param->num_proc + q] ;
				}
				common->prob[d] = realloc(common->prob[d], tot_num_prob * sizeof(double)) ;
				common->place_prob[d] = realloc(common->place_prob[d], tot_num_prob * sizeof(int)) ;
				
				MPI_Gatherv(MPI_IN_PLACE, 0, MPI_INT, common->place_prob[d], &num_prob_p[d*param->num_proc], displ_prob_p, MPI_INT, 0, MPI_COMM_WORLD) ;
				MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, common->prob[d], &num_prob_p[d*param->num_proc], displ_prob_p, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
				common->num_prob[d] = tot_num_prob ;
			}
		}
		if (!param->rank) {
			free(num_prob_p) ;
			free(displ_prob_p) ;
		}
	}
	
	// Combine scale factor information from all MPI ranks
	if (param->need_scaling && param->update_scale)
		MPI_Allreduce(MPI_IN_PLACE, common->psum_d, iter->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	if (mod->num_modes > 1)
		MPI_Allreduce(MPI_IN_PLACE, common->quat_norm, iter->tot_num_data*mod->num_modes, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	print_max_time("sync", "", param->verbosity > 1 && param->rank == 0) ;
	return avg_likelihood ;
}

void update_scale(struct max_data *common) {
	int d ;
	struct iterate *iter = common->iter ;
	struct params *param = iter->par ;

	for (d = 0 ; d < iter->tot_num_data ; ++d)
	if (!iter->blacklist[d])
		iter->scale[d] = iter->fcounts[d] / common->psum_d[d] ;

	print_max_time("scale", "", param->verbosity > 1 && param->rank == 0) ;
}

void free_max_data(struct max_data *data) {
	int detn, d ;
	struct iterate *iter = data->iter ;
	struct model *mod = iter->mod ;
	struct params *param = iter->par ;

	free(data->max_exp_p) ;
	free(data->info) ;
	free(data->likelihood) ;
	free(data->rmax) ;
	if (mod->num_modes > 1)
		free(data->quat_norm) ;
	if (data->prob[0] != NULL)
	for (d = 0 ; d < iter->tot_num_data ; ++d) {
		free(data->prob[d]) ;
		free(data->place_prob[d]) ;
	}
	free(data->prob) ;
	free(data->place_prob) ;
	free(data->num_prob) ;
	if (param->need_scaling && param->update_scale)
		free(data->psum_d) ;
	
	if (!data->within_openmp) {
		free(data->max_exp) ;
		free(data->u) ;
		free(data->p_norm) ;
		free(data->offset_prob) ;
	}
	else {
		for (detn = 0 ; detn < iter->num_det ; ++detn)
			free(data->all_views[detn]) ;
		free(data->all_views) ;
		free(data->model) ;
		free(data->weight) ;
		free(data->psum_r) ;
	}
	free(data) ;
}

// Other functions
int resparsify(double *vals, int *pos, int num_vals, double thresh) {
	int i, j ;
	
	for (i = 0 ; i < num_vals ; ++i) {
		if (vals[i] <= thresh) {
			num_vals-- ;
			for (j = i ; j < num_vals ; ++j) {
				vals[j] = vals[j+1] ;
				pos[j] = pos[j+1] ;
				pos[j+1] = -1 ;
			}
			i-- ;
		}
	}
	
	//vals = realloc(vals, num_vals*sizeof(double)) ;
	//pos = realloc(pos, num_vals*sizeof(int)) ;
	
	return num_vals ;
}

double calc_psum_r(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, d, curr_d, detn, rotind, mode, t, ind ;
	struct iterate *iter = common->iter ;
	struct dataset *frames = iter->dset ;
	struct model *mod = iter->mod ;
	struct quaternion *quat = iter->quat ;
	struct params *param = iter->par ;
	double temp, scalemax ;
	struct dataset *curr = frames ;
	double **prob = priv->prob ;
	int **place_prob = priv->place_prob, *num_prob = priv->num_prob ;
	
	scalemax = -DBL_MAX ;
	memset(priv->psum_r, 0, (iter->num_det)*sizeof(double)) ;
	rotind = (r*param->num_proc + param->rank) / param->num_modes ;
	mode = (r*param->num_proc + param->rank) % param->num_modes ;
	if (rotind >= quat->num_rot) {
		mode = r*param->num_proc + param->rank - param->num_modes * (quat->num_rot - 1) ;
		rotind = 0 ;
	}
	
	while (curr != NULL) {
		detn = iter->det_mapping[dset] ;
		for (curr_d = 0 ; curr_d < curr->num_data ; ++curr_d) {
			// Calculate frame number in full list
			d = curr->num_offset + curr_d ;

			// check if frame is blacklisted
			if (iter->blacklist[d])
				continue ;
			
			// For refinement, check if frame should be processed
			if (param->refine) {
				ind = -1 ;
				for (t = 0 ; t < iter->num_rel_quat[d] ; ++t)
				if (iter->quat_mapping[r*param->num_proc+param->rank] == iter->rel_quat[d][t]) {
					ind = t ;
					break ;
				}
				if (ind == -1)
					continue ;
			}
			
			// check if current frame has significant probability
			ind = -1 ;
			for (t = 0 ; t < num_prob[d] ; ++t)
			if (r*param->num_proc + param->rank == place_prob[d][t]) {
				ind = t ;
				break ;
			}
			if (ind == -1)
				continue ;
			
			// Exponentiate log-likelihood and normalize to get probabilities
			temp = prob[d][ind] ;
			if (curr->ftype < 2)
				prob[d][ind] = exp(iter->beta[d] * (prob[d][ind] - common->max_exp[d])) / common->p_norm[d] ; 
			else
				prob[d][ind] = exp(iter->beta[d] * (prob[d][ind] - common->max_exp[d]) / 2. / param->sigmasq) / common->p_norm[d] ;
			
			//if (param->need_scaling)
			//	priv->likelihood[d] += prob[d][ind] * (temp - iter->sum_fact[d] + frames->count[d]*log(iter->scale[d])) ;
			//else
			priv->likelihood[d] += prob[d][ind] * (temp - iter->sum_fact[d]) ;
			
			// Calculate denominator for update rule
			if (param->need_scaling) {
				priv->psum_r[detn] += prob[d][ind] * iter->scale[d] ;
				
				// Calculate denominator for scale factor update rule
				if (param->update_scale)
					priv->psum_d[d] += prob[d][ind] * common->u[detn][r] * iter->rescale[detn] ;
			}
			else
				priv->psum_r[detn] += prob[d][ind] ; 
			
			// Skip if probability is very low (saves time)
			if (prob[d][ind] < PROB_MIN)
				continue ;
			
			// If multiple modes, calculate occupancy of frame into each mode
			if (mod->num_modes > 1)
				priv->quat_norm[d*mod->num_modes + mode] += prob[d][ind] ;
			
			// Calculate mutual information of probability distribution
			if (mode >= param->num_modes)
				priv->info[d] += prob[d][ind] * log(prob[d][ind] / mod->num_modes) ;
			else
				priv->info[d] += prob[d][ind] * log(prob[d][ind] / quat->quats[rotind*5 + 4] * mod->num_modes) ;
			
			if (param->need_scaling && iter->scale[d] > scalemax)
				scalemax = iter->scale[d] ;
		}
		dset++ ;
		curr = curr->next ;
	}
	
	return scalemax ;
}

void print_max_time(char *pre_tag, char *post_tag, int flag) {
	if (!flag)
		return ;
	
	double diff ;
	double time_1 = tm1.tv_sec + tm1.tv_usec*1.e-6 ;
	double time_2 = tm2.tv_sec + tm2.tv_usec*1.e-6 ;
	
	if (time_1 > time_2) {
		gettimeofday(&tm2, NULL) ;
		diff = tm2.tv_sec + tm2.tv_usec*1.e-6 - time_1 ;
	}
	else {
		gettimeofday(&tm1, NULL) ;
		diff = tm1.tv_sec + tm1.tv_usec*1.e-6 - time_2 ;
	}
	
	fprintf(stderr, "\t%s\t%f %s\n", pre_tag, diff, post_tag) ;
}
