#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>
#include <gsl/gsl_sf_bessel.h>
#include "emc.h"

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
static void free_memory(struct max_data*) ;
// End Maximize

// Other functions
static int resparsify(double*, int*, int, double) ;
static double calc_psum_r(int, struct max_data*, struct max_data*) ;
static void update_tomogram_nobg(int, struct max_data*, struct max_data*) ;
static void gradient_rt(int, struct max_data*, double**, double**) ;
static void update_tomogram_bg(int, double, struct max_data*, struct max_data*) ;
void gradient_d(struct max_data*, uint8_t*, double*, double*) ;
void update_scale_bg(struct max_data*) ;
static void print_max_time(char*, char*, int) ;

double maximize() {
	double avg_likelihood ;
	gettimeofday(&tm1, NULL) ;
	struct max_data *common_data = malloc(sizeof(struct max_data)) ;
	common_data->within_openmp = 0 ;
	
	allocate_memory(common_data) ;
	calculate_rescale(common_data) ;

	#pragma omp parallel default(shared)
	{
		int r ;
		struct max_data *priv_data = malloc(sizeof(struct max_data)) ;
		
		priv_data->within_openmp = 1 ;
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
		
		free_memory(priv_data) ;
	}

	avg_likelihood = combine_information_mpi(common_data) ;
	if (param->need_scaling && param->update_scale)
		update_scale(common_data) ;
	
	if (!param->rank) {
		save_metrics(common_data) ;
		if (param->save_prob)
			save_prob(common_data) ;
	}
	print_max_time("save", "", param->rank == 0) ;
	free_memory(common_data) ;
	
	return avg_likelihood ;
}

void allocate_memory(struct max_data *data) {
	int detn, d ;
	
	// Both private and common
	data->rmax = calloc(frames->tot_num_data, sizeof(int)) ;
	data->info = calloc(frames->tot_num_data, sizeof(double)) ;
	data->likelihood = calloc(frames->tot_num_data, sizeof(double)) ;
	data->max_exp_p = malloc(frames->tot_num_data * sizeof(double)) ;
	for (d = 0 ; d < frames->tot_num_data ; ++d)
		data->max_exp_p[d] = MAX_EXP_START ;
	if (iter->modes > 1)
		data->quat_norm = calloc(iter->modes * frames->tot_num_data, sizeof(double)) ;
	
	data->prob = malloc(frames->tot_num_data * sizeof(double*)) ;
	data->place_prob = malloc(frames->tot_num_data * sizeof(int*)) ;
	for (d = 0 ; d < frames->tot_num_data ; ++d) {
		data->prob[d] = NULL ;
		data->place_prob[d] = NULL ;
	}
	data->num_prob = calloc(frames->tot_num_data, sizeof(int)) ;
	if (param->need_scaling && param->update_scale)
		data->psum_d = calloc(frames->tot_num_data, sizeof(double)) ;
		
	if (!data->within_openmp) { // common_data
		data->u = malloc(det[0].num_det * sizeof(double*)) ;
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
			data->u[detn] = calloc(quat->num_rot_p, sizeof(double)) ;
		data->max_exp = calloc(frames->tot_num_data, sizeof(double)) ;
		data->p_norm = calloc(frames->tot_num_data, sizeof(double)) ;
		data->offset_prob = calloc(frames->tot_num_data * omp_get_max_threads(), sizeof(int)) ;
		
		memset(iter->model2, 0, iter->modes*iter->vol*sizeof(double)) ;
		memset(iter->inter_weight, 0, iter->modes*iter->vol*sizeof(double)) ;
		print_max_time("alloc", "", param->rank == 0) ;
	}
	else { // priv_data
		data->all_views = malloc(det[0].num_det * sizeof(double*)) ;
		for (d = 0 ; d < det[0].num_det ; ++d)
			data->all_views[d] = malloc(det[d].num_pix * sizeof(double)) ;
		
		data->model = calloc(iter->modes*iter->vol, sizeof(double)) ;
		data->weight = calloc(iter->modes*iter->vol, sizeof(double)) ;
		
		data->psum_r = calloc(det[0].num_det, sizeof(double)) ;
		data->curr_ind = calloc(frames->tot_num_data, sizeof(int)) ;
		
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			data->prob[d] = malloc(4 * sizeof(double)) ;
			data->place_prob[d] = malloc(4 * sizeof(int)) ;
		}
		
		// Only for background-aware update
		if (det[0].with_bg && param->need_scaling) {
			data->mask = malloc(det[0].num_det * sizeof(uint8_t*)) ;
			data->G_old = malloc(det[0].num_det * sizeof(double*)) ;
			data->G_new = malloc(det[0].num_det * sizeof(double*)) ;
			data->G_latest = malloc(det[0].num_det * sizeof(double*)) ;
			data->W_old = malloc(det[0].num_det * sizeof(double*)) ;
			data->W_new = malloc(det[0].num_det * sizeof(double*)) ;
			data->W_latest = malloc(det[0].num_det * sizeof(double*)) ;
			for (detn = 0 ; detn < det[0].num_det ; ++detn) {
				data->mask[detn] = calloc(det[detn].num_pix, sizeof(uint8_t)) ;
				data->G_old[detn] = calloc(det[detn].num_pix, sizeof(double)) ;
				data->G_new[detn] = calloc(det[detn].num_pix, sizeof(double)) ;
				data->G_latest[detn] = calloc(det[detn].num_pix, sizeof(double)) ;
				data->W_old[detn] = calloc(det[detn].num_pix, sizeof(double)) ;
				data->W_new[detn] = calloc(det[detn].num_pix, sizeof(double)) ;
				data->W_latest[detn] = calloc(det[detn].num_pix, sizeof(double)) ;
			}
		}
	}
}

void calculate_rescale(struct max_data *data) {
	int detn ;
	double *total = calloc(det[0].num_det, sizeof(double)) ;
	char res_string[1024] = {'\0'}  ;
	
	// Calculate rescale factor by calculating mean model value over detector
	#pragma omp parallel default(shared)
	{
		int r, t, detn, mode, rotind ;
		double *priv_total = calloc(det[0].num_det, sizeof(double)) ;
		double **views = malloc(det[0].num_det * sizeof(double*)) ;
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
			views[detn] = malloc(det[detn].num_pix * sizeof(double)) ;
		
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < quat->num_rot_p ; ++r) {
			rotind = (r*param->num_proc + param->rank) / param->modes ;
			mode = (r*param->num_proc + param->rank) % param->modes ;
			if (rotind >= quat->num_rot) {
				mode = r*param->num_proc + param->rank - param->modes * (quat->num_rot - 1) ;
				rotind = 0 ;
			}
			//fprintf(stderr, "%d: %.3d - %.2d %.2d\n", omp_get_thread_num(), r, rotind, mode) ;
			
			for (detn = 0 ; detn < det[0].num_det ; ++detn) {
				// Second argument being 0. tells slice_gen to generate un-rescaled tomograms
				(*slice_gen)(&quat->quat[rotind*5], 0., views[detn], &iter->model1[mode*iter->vol], iter->size, &det[detn]) ;
				
				for (t = 0 ; t < det[detn].num_pix ; ++t)
				if (det[detn].mask[t] < 1)
					data->u[detn][r] += views[detn][t] ;
				
				priv_total[detn] += quat->quat[rotind*5 + 4] * data->u[detn][r] ;
			}
		}
		
		#pragma omp critical(total)
		{
			for (detn = 0 ; detn < det[0].num_det ; ++detn)
				total[detn] += priv_total[detn] ;
		}
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
			free(views[detn]) ;
		free(views) ;
		free(priv_total) ;
	}
	
	MPI_Allreduce(MPI_IN_PLACE, total, det[0].num_det, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	sprintf(res_string, "(=") ;
	for (detn = 0 ; detn < det[0].num_det ; ++detn) {
		iter->rescale[detn] = iter->mean_count[detn] / total[detn] * iter->modes ;
		sprintf(res_string + strlen(res_string), " %.6e", iter->rescale[detn]) ;
	}
	sprintf(res_string + strlen(res_string), ")") ;
	print_max_time("rescale", res_string, param->rank == 0) ;
	
	free(total) ;
}

void calculate_prob(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, t, d, curr_d, pixel, mode, rotind, detn, old_detn = -1, ind, new_num_prob ;
	struct dataset *curr = frames ;
	double pval, *view ;
	int *num_prob = priv->num_prob, **place_prob = priv->place_prob ;
	double **prob = priv->prob ;
	
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	mode = (r*param->num_proc + param->rank) % param->modes ;
	if (rotind >= quat->num_rot) {
		mode = r*param->num_proc + param->rank - param->modes * (quat->num_rot - 1) ;
		rotind = 0 ;
	}
	
	// Linked list of data sets from different files
	while (curr != NULL) {
		// Calculate slice for current detector
		detn = det[0].mapping[dset] ;
		view = priv->all_views[detn] ;
		if (detn != old_detn) {
			if (det[0].with_bg && param->need_scaling)
				(*slice_gen)(&quat->quat[rotind*5], 0., view, &iter->model1[mode*iter->vol], iter->size, &det[detn]) ;
			else
				(*slice_gen)(&quat->quat[rotind*5], 1., view, &iter->model1[mode*iter->vol], iter->size, &det[detn]) ;
		}
		old_detn = detn ;
		
		// For each frame in data set
		for (curr_d = 0 ; curr_d < curr->num_data ; ++curr_d) {
			// Calculate frame number in full list
			d = curr->num_data_prev + curr_d ;
			
			// check if frame is blacklisted
			if (frames->blacklist[d])
				continue ;
			
			// For refinement, check if frame should be processed
			if (param->refine) {
				ind = -1 ;
				for (t = 0 ; t < iter->num_rel_quat[d] ; ++t)
				if (iter->quat_mapping[rotind] == iter->rel_quat[d][t]) {
					ind = t ;
					break ;
				}
				if (ind == -1)
					continue ;
			}
			
			if (curr->type < 2) {
				// need_scaling is for if we want to assume variable incident intensity
				if (param->need_scaling && (param->iteration > 1 || param->known_scale))
					pval = log(quat->quat[rotind*5 + 4]) - common->u[detn][r] * iter->scale[d] ;
				else
					pval = log(quat->quat[rotind*5 + 4]) - common->u[detn][r] * iter->rescale[detn] ;
			}
			else {
				pval = 0. ;
			}
			
			if (curr->type == 0) {
				if (det[0].with_bg && param->need_scaling) {
					// For each pixel with one photon
					for (t = 0 ; t < curr->ones[curr_d] ; ++t) {
						pixel = curr->place_ones[curr->ones_accum[curr_d] + t] ;
						if (det[detn].mask[pixel] < 1)
							pval += log(view[pixel] * iter->scale[d] + iter->bgscale[d] * det[detn].background[pixel]) ;
					}
					
					// For each pixel with count_multi photons
					for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
						pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
						if (det[detn].mask[pixel] < 1)
							pval += curr->count_multi[curr->multi_accum[curr_d] + t] * log(view[pixel] * iter->scale[d] + iter->bgscale[d] * det[detn].background[pixel]) ;
					}
				}
				else {
					// For each pixel with one photon
					for (t = 0 ; t < curr->ones[curr_d] ; ++t) {
						pixel = curr->place_ones[curr->ones_accum[curr_d] + t] ;
						if (det[detn].mask[pixel] < 1)
							pval += view[pixel] ;
					}
					
					// For each pixel with count_multi photons
					for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
						pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
						if (det[detn].mask[pixel] < 1)
							pval += curr->count_multi[curr->multi_accum[curr_d] + t] * view[pixel] ;
					}
				}
			}
			else if (curr->type == 1) {
				for (t = 0 ; t < det[detn].num_pix ; ++t)
				if (det[detn].mask[t] < 1)
					pval += curr->int_frames[curr_d*curr->num_pix + t] * view[t] ;
			}
			else if (curr->type == 2) { // Gaussian EMC for double precision data without scaling
				for (t = 0 ; t < det[detn].num_pix ; ++t)
				if (det[detn].mask[t] < 1)
					pval -= pow(curr->frames[curr_d*curr->num_pix + t] - view[t]*iter->rescale[detn], 2.) ;
			}
			
			// Only save value in prob array if it is significant
			if (pval + PDIFF_THRESH/param->beta[d] > priv->max_exp_p[d]) {
				prob[d][num_prob[d]] = pval ;
				place_prob[d][num_prob[d]] = r*param->num_proc + param->rank ;
				num_prob[d]++ ;
				
				// If num_prob is a power of two, expand array
				if (num_prob[d] >= 4 && num_prob[d] < quat->num_rot_p && (num_prob[d] & (num_prob[d] - 1)) == 0) {
					new_num_prob = num_prob[d] * 2 < quat->num_rot_p ? num_prob[d] * 2 : quat->num_rot_p ;
					prob[d] = realloc(prob[d], new_num_prob * sizeof(double)) ;
					place_prob[d] = realloc(place_prob[d], new_num_prob * sizeof(int)) ;
				}
			}
			
			// Note maximum log-likelihood for each frame among 'r's tested by this MPI rank and OMP rank
			// Recalculate sparse array with new maximum
			if (pval > priv->max_exp_p[d]) {
				priv->max_exp_p[d] = pval ;
				priv->rmax[d] = r*param->num_proc + param->rank ;
				num_prob[d] = resparsify(prob[d], place_prob[d], num_prob[d], pval - PDIFF_THRESH/param->beta[d]) ;
			}
		}
		
		curr = curr->next ;
		dset++ ;
	}
	
	if ((r*param->num_proc + param->rank)%(quat->num_rot * param->modes / 10) == 0)
		fprintf(stderr, "\t\tFinished r = %d/%d\n", r*param->num_proc + param->rank, quat->num_rot * param->modes + param->nonrot_modes) ;
	print_max_time("prob", "", (r == quat->num_rot_p-1) && (param->rank == 0)) ;
}

void normalize_prob(struct max_data *priv, struct max_data *common) {
	int r, d, omp_rank = omp_get_thread_num() ;
	double *priv_norm = calloc(frames->tot_num_data, sizeof(double)) ;
	
	// Calculate max_log_prob over all OpenMP ranks (and the r for that maximum)
	#pragma omp critical(maxexp)
	{
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (priv->max_exp_p[d] > common->max_exp_p[d]) {
			common->max_exp_p[d] = priv->max_exp_p[d] ;
			common->rmax[d] = priv->rmax[d] ;
		}
	}
	#pragma omp barrier
	
	if (omp_rank == 0) {
		MPI_Allreduce(common->max_exp_p, common->max_exp, frames->tot_num_data, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) ;
		
		// Determine 'r' for which log-likelihood is maximum
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (common->max_exp[d] != common->max_exp_p[d] || common->max_exp_p[d] == MAX_EXP_START)
			common->rmax[d] = -1 ;
		
		MPI_Allreduce(MPI_IN_PLACE, common->rmax, frames->tot_num_data, MPI_INT, MPI_MAX, MPI_COMM_WORLD) ;
	}
	#pragma omp barrier
	
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	for (r = 0 ; r < priv->num_prob[d] ; ++r) 
	if (frames->type < 2)
		priv_norm[d] += exp(param->beta[d] * (priv->prob[d][r] - common->max_exp[d])) ;
	else
		priv_norm[d] += exp(param->beta[d] * (priv->prob[d][r] - common->max_exp[d]) / 2. / param->sigmasq) ;
	
	#pragma omp critical(psum)
	{
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			common->p_norm[d] += priv_norm[d] ;
	}
	#pragma omp barrier
	
	if (omp_rank == 0)
		MPI_Allreduce(MPI_IN_PLACE, common->p_norm, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	#pragma omp barrier
	
	free(priv_norm) ;
	print_max_time("norm", "", param->rank == 0 && omp_rank == 0) ;
}

void update_tomogram(int r, struct max_data *priv, struct max_data *common) {
	double scalemax ;
	
	scalemax = calc_psum_r(r, priv, common) ;
	
	if (det[0].with_bg && param->need_scaling)
		update_tomogram_bg(r, scalemax, priv, common) ;
	else
		update_tomogram_nobg(r, priv, common) ;
}

void merge_tomogram(int r, struct max_data *priv) {
	int detn, mode, rotind ;
	
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	mode = (r*param->num_proc + param->rank) % param->modes ;
	if (rotind >= quat->num_rot) {
		mode = r*param->num_proc + param->rank - param->modes * (quat->num_rot - 1) ;
		rotind = 0 ;
	}
	
	// If no data frame has any probability for this orientation, don't merge
	for (detn = 0 ; detn < det[0].num_det ; ++detn)
	if (priv->psum_r[detn] > 0.)
		(*slice_merge)(&quat->quat[rotind*5], priv->all_views[detn], &priv->model[mode*iter->vol], &priv->weight[mode*iter->vol], iter->size, &det[detn]) ;
	
	if ((r*param->num_proc + param->rank)%(quat->num_rot * param->modes / 10) == 0)
		fprintf(stderr, "\t\tFinished r = %d/%d\n", r*param->num_proc + param->rank, quat->num_rot * param->modes + param->nonrot_modes) ;
}

void combine_information_omp(struct max_data *priv, struct max_data *common) {
	int d, r, omp_rank = omp_get_thread_num() ;
	int nthreads = omp_get_num_threads() ;
	long x ;
	
	print_max_time("update", "", param->rank == 0 && omp_rank == 0) ;
	 
	#pragma omp critical(model)
	{
		for (x = 0 ; x < iter->modes * iter->vol ; ++x) {
			iter->model2[x] += priv->model[x] ;
			iter->inter_weight[x] += priv->weight[x] ;
		}
	}
	
	#pragma omp critical(like_info)
	{
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			common->likelihood[d] += priv->likelihood[d] ;
			common->info[d] += priv->info[d] ;
		}
	}
	
	if (param->need_scaling && param->update_scale) {
		#pragma omp critical(scale)
		{
			for (d = 0 ; d < frames->tot_num_data ; ++d)
			if (!frames->blacklist[d])
				common->psum_d[d] += priv->psum_d[d] ;
		}
	}
	
	if (iter->modes > 1) {
		#pragma omp critical(quat_norm)
		for (d = 0 ; d < frames->tot_num_data * iter->modes ; ++d)
			common->quat_norm[d] += priv->quat_norm[d] ;
	}
	print_max_time("osync", "", param->rank == 0 && omp_rank == 0) ;
	
	// Only calculate common probabilities if needed for update_scale or to save
	if ((param->need_scaling && param->update_scale && det[0].with_bg) || (param->save_prob)) {
		// Calculate offsets to combine sparse probabilities for each OpenMP rank
		#pragma omp critical(offset_prob)
		{
			for (d = 0 ; d < frames->tot_num_data ; ++d) {
				common->num_prob[d] += priv->num_prob[d] ;
				for (r = omp_rank + 1 ; r < nthreads ; ++r)
					common->offset_prob[d*nthreads + r] += priv->num_prob[d] ;
			}
		}
		#pragma omp barrier
			
		// Allocate common prob arrays
		#pragma omp for schedule(static,1)
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			common->prob[d] = malloc(common->num_prob[d] * sizeof(double)) ;
			common->place_prob[d] = malloc(common->num_prob[d] * sizeof(int)) ;
		}
		
		// Populate common->prob array for all d
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		for (r = 0 ; r < priv->num_prob[d] ; ++r) {
			common->prob[d][common->offset_prob[d*nthreads + omp_rank] + r] = priv->prob[d][r] ;
			common->place_prob[d][common->offset_prob[d*nthreads + omp_rank] + r] = priv->place_prob[d][r] ;
		}
		#pragma omp barrier
		
		// Sparsify probs based on PROB_MIN threshold
		#pragma omp for schedule(static,1)
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			common->num_prob[d] = resparsify(common->prob[d], common->place_prob[d], common->num_prob[d], PROB_MIN) ;
	}
	print_max_time("ocprob", "", param->rank == 0 && omp_rank == 0) ;
}

double combine_information_mpi(struct max_data *common) {
	int d ;
	double avg_likelihood = 0. ;
	iter->mutual_info = 0. ;
	
	// Combine 3D volumes from all MPI ranks
	if (param->rank) {
		MPI_Reduce(iter->model2, iter->model2, iter->modes*iter->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
		MPI_Reduce(iter->inter_weight, iter->inter_weight, iter->modes*iter->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
	}
	else {
		MPI_Reduce(MPI_IN_PLACE, iter->model2, iter->modes*iter->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
		MPI_Reduce(MPI_IN_PLACE, iter->inter_weight, iter->modes*iter->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
	}
	
	// Combine mutual info and likelihood from all MPI ranks
	MPI_Allreduce(MPI_IN_PLACE, common->likelihood, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	MPI_Allreduce(MPI_IN_PLACE, common->info, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (!frames->blacklist[d]) {
		iter->mutual_info += common->info[d] ;
		avg_likelihood += common->likelihood[d] ;
	}
	
	iter->mutual_info /= (frames->tot_num_data - frames->num_blacklist) ;
	avg_likelihood /= (frames->tot_num_data - frames->num_blacklist) ;
	
	// Combine scale factor information from all MPI ranks
	if (param->need_scaling && param->update_scale)
		MPI_Allreduce(MPI_IN_PLACE, common->psum_d, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	if (iter->modes > 1)
		MPI_Allreduce(MPI_IN_PLACE, common->quat_norm, frames->tot_num_data*iter->modes, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	print_max_time("isync", "", param->rank == 0) ;
	
	// Only calculate common probabilities if needed for update_scale or to save
	if ((param->need_scaling && param->update_scale && det[0].with_bg) || (param->save_prob)) {
		int p, q, tot_num_prob ;
		int *num_prob_p = calloc(frames->tot_num_data * param->num_proc, sizeof(int)) ;
		int *displ_prob_p = NULL ;
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			num_prob_p[d*param->num_proc + param->rank] = common->num_prob[d] ;
		if (param->rank) {
			MPI_Reduce(num_prob_p, NULL, frames->tot_num_data*param->num_proc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) ;
			free(num_prob_p) ;
		}
		else {
			MPI_Reduce(MPI_IN_PLACE, num_prob_p, frames->tot_num_data*param->num_proc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) ;
			displ_prob_p = malloc(param->num_proc * sizeof(int)) ;
		}
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
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
	
	print_max_time("icprob", "", param->rank == 0) ;
	return avg_likelihood ;
}

void update_scale(struct max_data *common) {
	if (det[0].with_bg) {
		update_scale_bg(common) ;
	}
	else {
		int d ;
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (!frames->blacklist[d])
			iter->scale[d] = frames->count[d] / common->psum_d[d] ;
		print_max_time("scale", "", param->rank == 0) ;
	}
}

void free_memory(struct max_data *data) {
	int detn, d ;
	free(data->max_exp_p) ;
	free(data->info) ;
	free(data->likelihood) ;
	free(data->rmax) ;
	if (iter->modes > 1)
		free(data->quat_norm) ;
	if (data->prob[0] != NULL)
	for (d = 0 ; d < frames->tot_num_data ; ++d) {
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
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
			free(data->u[detn]) ;
		free(data->u) ;
		free(data->p_norm) ;
		free(data->offset_prob) ;
	}
	else {
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
			free(data->all_views[detn]) ;
		free(data->all_views) ;
		free(data->model) ;
		free(data->weight) ;
		free(data->psum_r) ;
		free(data->curr_ind) ;
		if (det[0].with_bg && param->need_scaling) {
			for (detn = 0 ; detn < det[0].num_det ; ++detn) {
				free(data->mask[detn]) ;
				free(data->G_old[detn]) ;
				free(data->G_new[detn]) ;
				free(data->G_latest[detn]) ;
				free(data->W_old[detn]) ;
				free(data->W_new[detn]) ;
				free(data->W_latest[detn]) ;
			}
			free(data->mask) ;
			free(data->G_old) ;
			free(data->G_new) ;
			free(data->G_latest) ;
			free(data->W_old) ;
			free(data->W_new) ;
			free(data->W_latest) ;
		}
	}
	free(data) ;
}


// Other functions
int resparsify(double *vals, int *pos, int num_vals, double thresh) {
	int nv = 0;
	for (int i = 0 ; i < num_vals ; ++i) {
		if (vals[i] > thresh) {
			vals[nv] = vals[i];
			pos[nv] = pos[i];
 			nv++;
		}
	}
	return nv;
}


double calc_psum_r(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, d, curr_d, detn, rotind, mode, t, ind, true_r ;
	double temp, scalemax ;
	struct dataset *curr = frames ;
	double **prob = priv->prob ;
	int **place_prob = priv->place_prob, *num_prob = priv->num_prob ;
	
	scalemax = -DBL_MAX ;
	memset(priv->psum_r, 0, (det[0].num_det)*sizeof(double)) ;
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	mode = (r*param->num_proc + param->rank) % param->modes ;
	if (rotind >= quat->num_rot) {
		mode = r*param->num_proc + param->rank - param->modes * (quat->num_rot - 1) ;
		rotind = 0 ;
	}
	
	while (curr != NULL) {
		detn = det[0].mapping[dset] ;
		for (curr_d = 0 ; curr_d < curr->num_data ; ++curr_d) {
			// Calculate frame number in full list
			d = curr->num_data_prev + curr_d ;

			// check if frame is blacklisted
			if (frames->blacklist[d])
				continue ;
			
			// For refinement, check if frame should be processed
			if (param->refine) {
				ind = -1 ;
				for (t = 0 ; t < iter->num_rel_quat[d] ; ++t)
				if (iter->quat_mapping[rotind] == iter->rel_quat[d][t]) {
					ind = t ;
					break ;
				}
				if (ind == -1)
					continue ;
			}
			
			// check if current frame has significant probability
			true_r = r*param->num_proc + param->rank ;
			if (true_r == place_prob[d][priv->curr_ind[d]]) ;
			else if (priv->curr_ind[d] < num_prob[d] - 1 && true_r == place_prob[d][priv->curr_ind[d] + 1])
				priv->curr_ind[d]++ ;
			else
				continue ;
			ind = priv->curr_ind[d] ;
			
			// Exponentiate log-likelihood and normalize to get probabilities
			temp = prob[d][ind] ;
			if (frames->type < 2)
				prob[d][ind] = exp(param->beta[d] * (prob[d][ind] - common->max_exp[d])) / common->p_norm[d] ; 
			else
				prob[d][ind] = exp(param->beta[d] * (prob[d][ind] - common->max_exp[d]) / 2. / param->sigmasq) / common->p_norm[d] ;
			
			//if (param->need_scaling)
			//	priv->likelihood[d] += prob[d][ind] * (temp - frames->sum_fact[d] + frames->count[d]*log(iter->scale[d])) ;
			//else
			priv->likelihood[d] += prob[d][ind] * (temp - frames->sum_fact[d]) ;
			
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
			if (iter->modes > 1)
				priv->quat_norm[d*iter->modes + mode] += prob[d][ind] ;
			
			// Calculate mutual information of probability distribution
			if (mode >= param->modes)
				priv->info[d] += prob[d][ind] * log(prob[d][ind] / iter->modes) ;
			else
				priv->info[d] += prob[d][ind] * log(prob[d][ind] / quat->quat[rotind*5 + 4] * iter->modes) ;
			
			if (param->need_scaling && iter->scale[d] > scalemax)
				scalemax = iter->scale[d] ;
		}
		dset++ ;
		curr = curr->next ;
	}
	
	return scalemax ;
}

void update_tomogram_nobg(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, t, d, curr_d, pixel, detn, ind, rotind ;
	double *view ;
	struct dataset *curr = frames ;
	double **prob = priv->prob ;
	int **place_prob = priv->place_prob ;
	
	for (detn = 0 ; detn < det[0].num_det ; ++detn) 
		memset(priv->all_views[detn], 0, det[detn].num_pix*sizeof(double)) ;
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	
	while (curr != NULL) {
		// Calculate slice for current detector
		detn = det[0].mapping[dset] ;
		view = priv->all_views[detn] ;
		
		for (curr_d = 0 ; curr_d < curr->num_data ; ++curr_d) {
			// Calculate frame number in full list
			d = curr->num_data_prev + curr_d ;
			
			// check if frame is blacklisted
			if (frames->blacklist[d])
				continue ;
			
			// For refinement, check if frame should be processed
			if (param->refine) {
				ind = -1 ;
				for (t = 0 ; t < iter->num_rel_quat[d] ; ++t)
				if (iter->quat_mapping[rotind] == iter->rel_quat[d][t]) {
					ind = t ;
					break ;
				}
				if (ind == -1)
					continue ;
			}
			
			// check if current frame has significant probability
			if (r*param->num_proc + param->rank != place_prob[d][priv->curr_ind[d]])
				continue ;
			ind = priv->curr_ind[d] ;
			
			// Skip if probability is very low (saves time)
			if (!(prob[d][ind] > PROB_MIN))
				continue ;
			
			if (curr->type == 0) {
				// For all pixels with one photon
				for (t = 0 ; t < curr->ones[curr_d] ; ++t) {
					pixel = curr->place_ones[curr->ones_accum[curr_d] + t] ;
					if (det[detn].mask[pixel] < 2)
						view[pixel] += prob[d][ind] ;
				}
				
				// For all pixels with count_multi photons
				for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
					pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
					if (det[detn].mask[pixel] < 2)
						view[pixel] += curr->count_multi[curr->multi_accum[curr_d] + t] * prob[d][ind] ;
				}
			}
			else if (curr->type == 1) {
				for (t = 0 ; t < curr->num_pix ; ++t)
					view[t] += curr->int_frames[curr_d*curr->num_pix + t] * prob[d][ind] ;
			}
			else if (curr->type == 2) { // Gaussian EMC update without scaling
				for (t = 0 ; t < curr->num_pix ; ++t)
				if (det[detn].mask[t] < 2)
					view[t] += curr->frames[curr_d*curr->num_pix + t] * prob[d][ind] ;
			}
		}
		
		curr = curr->next ;
		dset++ ;
	}
	
	for (detn = 0 ; detn < det[0].num_det ; ++detn) 
	for (t = 0 ; t < det[detn].num_pix ; ++t)
	if (priv->psum_r[detn] > 0.)
		priv->all_views[detn][t] /= priv->psum_r[detn] ;
}

void gradient_rt(int r, struct max_data *priv, double **views, double **gradients) {
	int dset = 0, t, d, curr_d, pixel, detn, ind, rotind ;
	double val, *grad, *view ;
	struct dataset *curr = frames ;
	int **place_prob = priv->place_prob ;
	double **prob = priv->prob ;
	
	// Initialization:
	// 	grad = -sum_d P_dr * phi_d
	// 	mask = 1
	for (detn = 0 ; detn < det[0].num_det ; ++detn) {
		grad = gradients[detn] ;
		view = views[detn] ;
		
		for (t = 0 ; t < det[detn].num_pix ; ++t) {
			if (priv->mask[detn][t] < 128) {
				grad[t] = - priv->psum_r[detn] ;
				priv->mask[detn][t] = 1 ;
			}
			else if (priv->mask[detn][t] == 160) {
				grad[t] = -DBL_MAX ;
			}
			else {
				grad[t] = 0. ;
			}
		}
	}
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	
	while (curr != NULL) {
		detn = det[0].mapping[dset] ;
		grad = gradients[detn] ;
		view = views[detn] ;
		
		for (curr_d = 0 ; curr_d < curr->num_data ; ++curr_d) {
			// Calculate frame number in full list
			d = curr->num_data_prev + curr_d ;
			
			// check if frame is blacklisted
			if (frames->blacklist[d])
				continue ;
			
			// For refinement, check if frame should be processed
			if (param->refine) {
				ind = -1 ;
				for (t = 0 ; t < iter->num_rel_quat[d] ; ++t)
				if (iter->quat_mapping[rotind] == iter->rel_quat[d][t]) {
					ind = t ;
					break ;
				}
				if (ind == -1)
					continue ;
			}
			
			// check if current frame has significant probability
			if (r*param->num_proc + param->rank != place_prob[d][priv->curr_ind[d]])
				continue ;
			ind = priv->curr_ind[d] ;
			
			// Skip if probability is very low (saves time)
			if (!(prob[d][ind] > PROB_MIN))
				continue ;
			
			// Currently only working with type-0 data
			// For each pixel with one photon
			for (t = 0 ; t < curr->ones[curr_d] ; ++t) {
				pixel = curr->place_ones[curr->ones_accum[curr_d] + t] ;
				if (priv->mask[detn][pixel] < 128) {
					val = view[pixel] * iter->scale[d] + iter->bgscale[d] * det[detn].background[pixel] ;
					grad[pixel] += prob[d][ind] * iter->scale[d] / val ;
					priv->mask[detn][pixel] = 0 ;
				}
				else if (priv->mask[detn][pixel] == 128) {
					grad[pixel] += prob[d][ind] ;
				}
				else if (priv->mask[detn][pixel] == 160) {
					grad[pixel] = fmax(grad[pixel], iter->scale[d] / iter->bgscale[d]) ;
				}
			}
			
			// For each pixel with count_multi photons
			for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
				pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
				if (priv->mask[detn][pixel] < 128) {
					val = view[pixel] * iter->scale[d] + iter->bgscale[d] * det[detn].background[pixel] ;
					grad[pixel] += prob[d][ind] * curr->count_multi[curr->multi_accum[curr_d] + t] * iter->scale[d] / val ;
					priv->mask[detn][pixel] = 0 ;
				}
				else if (priv->mask[detn][pixel] == 128) {
					grad[pixel] += prob[d][ind] * curr->count_multi[curr->multi_accum[curr_d] + t] ;
				}
				else if (priv->mask[detn][pixel] == 160) {
					grad[pixel] = fmax(grad[pixel], iter->scale[d] / iter->bgscale[d]) ;
				}
			}
		}
		
		dset++ ;
		curr = curr->next ;
	}
}

void update_tomogram_bg(int r, double scalemax, struct max_data *priv, struct max_data *common) {
	int i, t, detn ;
	int nmask, tot_num_pix = 0 ;
	double val ;
	//double x0, x1, f0, f1, f2 ; // For Ridder's method
	double grad_tol = 1.e-4 ;
	
	nmask = 0 ;
	for (detn = 0 ; detn < det[0].num_det ; ++detn) {
		memset(priv->all_views[detn], 0, det[detn].num_pix*sizeof(double)) ;
		memset(priv->W_old[detn], 0, det[detn].num_pix*sizeof(double)) ;
		memset(priv->W_new[detn], 0, det[detn].num_pix*sizeof(double)) ;
		memset(priv->W_latest[detn], 0, det[detn].num_pix*sizeof(double)) ;
		tot_num_pix += det[detn].num_pix ;
		
		for (t = 0 ; t < det[detn].num_pix ; ++t) {
			if (det[detn].mask[t] < 2) {
				if (det[detn].background[t] / scalemax < 1.e-2 * det[detn].powder[t]) {
					priv->mask[detn][t] = 128 ;
				}
				else {
					priv->W_new[detn][t] = 1.e-8 - det[detn].background[t] / scalemax ;
					priv->mask[detn][t] = 0 ;
				}
			}
			else {
				priv->mask[detn][t] = 255 ;
			}
		}
	}
	
	// mask values:
	//   0 : Still to be optimized
	// 128 : Negligible background (can do standard update)
	// 160 : Signal to gradient_rt() to calculate scale_max rather than gradient
	// 192 : Had to adjust search window to have opposite signs
	// 255 : Optimal pixel
	// Set search bounds
	// Calculate G(0) and check sign
	gradient_rt(r, priv, priv->W_old, priv->G_old) ;
	for (detn = 0 ; detn < det[0].num_det ; ++detn)
	for (t = 0 ; t < det[detn].num_pix ; ++t) {
		if (fabs(priv->G_old[detn][t]) < grad_tol) {
			priv->all_views[detn][t] = 0. ;
			priv->mask[detn][t] = 255 ;
		}
		else if (priv->G_old[detn][t] > 0.) {
			val = frames->tot_mean_count / det[detn].num_pix ; // Average photons/pixel
			priv->W_new[detn][t] = det[detn].powder[t] > val ? det[detn].powder[t] : val ; // Check against powder sum value
			priv->W_new[detn][t] *= 2. ; // Double
		}
	}
	
	// Find far end of search window
	for (i = 0 ; i < 10 ; ++i) {
		nmask = 0 ;
		gradient_rt(r, priv, priv->W_new, priv->G_new) ;
		
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
		for (t = 0 ; t < det[detn].num_pix ; ++t) {
			if (priv->mask[detn][t] == 0) {
				if (priv->G_old[detn][t] < 0. && priv->G_new[detn][t] < 0.) {      // If both negative, calculate scale_max
					priv->mask[detn][t] = 160 ; // Forces gradient_rt to calculate scale_max
				}
				else if (priv->G_old[detn][t] > 0. && priv->G_new[detn][t] > 0.) { // If both positive, double W_new
					priv->W_new[detn][t] *= 2 ;
				}
				else {                                                             // If opposite signs, stop updating
					priv->mask[detn][t] = 192 ;
					nmask++ ;
				}
			}
			else if (priv->mask[detn][t] == 160) { // Set W_new to be just above minimum background
				priv->W_new[detn][t] = 1.e-8 - det[detn].background[t] / priv->G_new[detn][t] ;
				priv->mask[detn][t] = 0 ;
			}
			else {
				nmask++ ;
			}
		}
		
		if (nmask == tot_num_pix)
			break ;
	}
	if (i == 10 && nmask/((double)tot_num_pix) < 0.9)
		fprintf(stderr, "%.5d bad search bounds, %d/%d\n", r*param->num_proc + param->rank, nmask, tot_num_pix) ;
	
	// Bounded root-finding using bisection/regula falsi/Ridder's
	for (i = 0 ; i < 50 ; ++i) { // Doing 50 iterations
		nmask = 0 ;
		// Update value of W_latest
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
		for (t = 0 ; t < det[detn].num_pix ; ++t) {
			if (priv->mask[detn][t] == 255) {      // Already optimal
				nmask++ ;
			}
			else if (priv->mask[detn][t] == 1) {   // No photon at pixel
				priv->mask[detn][t] = 255 ;
				priv->all_views[detn][t] = 0. ;
				nmask++ ;
			}
			else if (priv->mask[detn][t] == 128) { // Negligible background
				priv->all_views[detn][t] = priv->G_old[detn][t] / priv->psum_r[detn] ;
				priv->mask[detn][t] = 255 ;
				nmask++ ;
			}
			else if (priv->mask[detn][t] == 192) { // W_new had to be adjusted when finding search window
				priv->mask[detn][t] = 0 ;
				priv->W_latest[detn][t] = 0.5 * (priv->W_old[detn][t] + priv->W_new[detn][t]) ;
				/*
				if (i % 2 == 0) {
					priv->W_latest[detn][t] = 0.5 * (priv->W_old[detn][t] + priv->W_new[detn][t]) ;
				}
				else {
					x0 = priv->W_old[detn][t] ;
					x1 = priv->W_latest[detn][t] ;
					f0 = priv->G_old[detn][t] ;
					f1 = priv->G_latest[detn][t] ;
					f2 = priv->G_new[detn][t] ;
					priv->W_latest[detn][t] = x1 + (x1-x0)*copysign(1, f0)*f1/sqrt(f1*f1 - f0*f2) ;
				}
				*/
			}
			else if (priv->mask[detn][t] == 0) {   // Searching for root
				// Regula falsi (secant) update
				//priv->W_latest[detn][t] = (priv->W_old[detn][t]*priv->G_new[detn][t] - priv->W_new[detn][t]*priv->G_old[detn][t]) / (priv->G_new[detn][t] - priv->G_old[detn][t]) ;
				// Bisection update
				priv->W_latest[detn][t] = 0.5 * (priv->W_old[detn][t] + priv->W_new[detn][t]) ;
				// Ridder's update
				/*
				if (i % 2 == 0) {
					priv->W_latest[detn][t] = 0.5 * (priv->W_old[detn][t] + priv->W_new[detn][t]) ;
				}
				else {
					x0 = priv->W_old[detn][t] ;
					x1 = priv->W_latest[detn][t] ;
					f0 = priv->G_old[detn][t] ;
					f1 = priv->G_latest[detn][t] ;
					f2 = priv->G_new[detn][t] ;
					priv->W_latest[detn][t] = x1 + (x1-x0)*copysign(1, f0)*f1/sqrt(f1*f1 - f0*f2) ;
				}
				*/
			}
		}
		
		// Calculate G_latest(W_latest)
		gradient_rt(r, priv, priv->W_latest, priv->G_latest) ;
		
		// Test for convergence and change search window
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
		for (t = 0 ; t < det[detn].num_pix ; ++t)
		if (priv->mask[detn][t] < 255) {
			if (fabs(priv->G_latest[detn][t]) < grad_tol) {                // Converged
				priv->all_views[detn][t] = priv->W_latest[detn][t] ;
				priv->mask[detn][t] = 255 ;
				nmask++ ;
			}
			//else if (i % 2 == 1 && priv->G_latest[detn][t] * priv->G_old[detn][t] > 0) {
			else if (priv->G_latest[detn][t] * priv->G_old[detn][t] > 0) { // Shift window to W_old
				priv->W_old[detn][t] = priv->W_latest[detn][t] ;
				priv->G_old[detn][t] = priv->G_latest[detn][t] ;
			}
			//else if (i % 2 == 1) {
			else {                                                         // Shift window towards W_new
				priv->W_new[detn][t] = priv->W_latest[detn][t] ;
				priv->G_new[detn][t] = priv->G_latest[detn][t] ;
			}
		}
		
		if (nmask == tot_num_pix)
			break ;
	}
	if (i == 50 && nmask/((double)tot_num_pix) < 0.9)
		fprintf(stderr, "%.5d not converged, %d/%d\n", r*param->num_proc + param->rank, nmask, tot_num_pix) ;
}

void gradient_d(struct max_data *common, uint8_t *mask, double *scale, double *grad) {
	int d ;
	 
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (mask[d] == 0)
		grad[d] = 0. ;
	
	#pragma omp parallel default(shared)
	{
		int r, d, t, detn, curr_d, pixel, rotind, mode, ind, dset ;
		double val ;
		double *view, **views = malloc(det[0].num_det * sizeof(double*)) ;
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
			views[detn] = malloc(det[detn].num_pix * sizeof(double)) ;
		double *priv_grad = calloc(frames->tot_num_data, sizeof(double)) ;
		struct dataset *curr ;
		
		#pragma omp for schedule(static,1)
	 	for (r = 0 ; r < quat->num_rot_p ; ++r) {
			rotind = (r*param->num_proc + param->rank) / param->modes ;
			mode = (r*param->num_proc + param->rank) % param->modes ;
			if (rotind >= quat->num_rot) {
				mode = r*param->num_proc + param->rank - param->modes * (quat->num_rot - 1) ;
				rotind = 0 ;
			}
			for (detn = 0 ; detn < det[0].num_det ; ++detn) {
				(*slice_gen)(&quat->quat[rotind*5], 0., views[detn], &iter->model1[mode*iter->vol], iter->size, &det[detn]) ;
				for (t = 0 ; t < det[detn].num_pix ; ++t)
					views[detn][t] *= iter->rescale[detn] ;
			}
			curr = frames ;
			dset = 0 ;
			
			while (curr != NULL) {
				detn = det[0].mapping[dset] ;
				view = views[detn] ;
				for (curr_d = 0 ; curr_d < curr->num_data ; ++curr_d) {
					d = curr->num_data_prev + curr_d ;
					
					if (mask[d] > 0)
						continue ;
					
					// check if current frame has significant probability
					ind = -1 ;
					for (t = 0 ; t < common->num_prob[d] ; ++t)
					if (r*param->num_proc + param->rank == common->place_prob[d][t]) {
						ind = t ;
						break ;
					}
					if (ind == -1)
						continue ;
					
					// For each pixel with one photon
					for (t = 0 ; t < curr->ones[curr_d] ; ++t) {
						pixel = curr->place_ones[curr->ones_accum[curr_d] + t] ;
						if (det[detn].mask[pixel] < 1) { // Use only relevant pixels
						//if (det[detn].mask[pixel] < 2) { // Exclude bad pixels
							val = view[pixel] * scale[d] + iter->bgscale[d] * det[detn].background[pixel] ;
							priv_grad[d] += common->prob[d][ind] * view[pixel] / val ;
						}
					}
					
					// For each pixel with count_multi photons
					for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
						pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
						if (det[detn].mask[pixel] < 1) { // Use only relevant pixels
						//if (det[detn].mask[pixel] < 2) { // Exclude bad pixels
							val = view[pixel] * scale[d] + iter->bgscale[d] * det[detn].background[pixel] ;
							priv_grad[d] += common->prob[d][ind] * curr->count_multi[curr->multi_accum[curr_d] + t] * view[pixel] / val ;
						}
					}
				}
				
				dset++ ;
				curr = curr->next ;
			}
		}
		
		#pragma omp critical(grad)
		{
			for (d = 0 ; d < frames->tot_num_data ; ++d)
				grad[d] += priv_grad[d] ;
		}
		
		free(priv_grad) ;
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
			free(views[detn]) ;
		free(views) ;
	}
	
	MPI_Allreduce(MPI_IN_PLACE, grad, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (mask[d] == 0)
		grad[d] -= common->psum_d[d] ;
}

void update_scale_bg(struct max_data *common) {
	int d, i, num_mask ;
	//double x0, x1, f0, f1, f2 ; // For Ridders method
	double *scale_old = calloc(frames->tot_num_data, sizeof(double)) ;
	double *scale_new = calloc(frames->tot_num_data, sizeof(double)) ;
	double *scale_latest = calloc(frames->tot_num_data, sizeof(double)) ;
	double *Gd_old = calloc(frames->tot_num_data, sizeof(double)) ;
	double *Gd_new = calloc(frames->tot_num_data, sizeof(double)) ;
	double *Gd_latest = calloc(frames->tot_num_data, sizeof(double)) ;
	uint8_t *mask = calloc(frames->tot_num_data, sizeof(uint8_t)) ;
	for (d = 0 ; d < frames->tot_num_data ; ++d) {
		if (frames->blacklist[d]) {
			mask[d] = 255 ;
			iter->scale[d] = -1. ;
			continue ;
		}
		
		if (param->iteration > 1 || param->known_scale)
			scale_new[d] = 4. * iter->scale[d] ;
		else
			scale_new[d] = 32. ;
	}
	
	// Set search bounds
	// 	Calculate G(0)
	gradient_d(common, mask, scale_old, Gd_old) ;
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (mask[d] != 255) {
		if (fabs(Gd_old[d]) < 1.e-6) {
			iter->scale[d] = 0. ;
			mask[d] = 255 ;
		}
		else if (Gd_old[d] < 0.) { // TODO Handle this better
			//iter->scale[d] = 0. ;
			mask[d] = 255 ; // Leave iter->scale unchanged
		}
	}
	
	// 	Calculate phi_max and G(phi_max)
	for (i = 0 ; i < 5 ; ++i) {
		gradient_d(common, mask, scale_new, Gd_new) ;
		num_mask = 0 ;
		
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			if (mask[d] != 0) {
				num_mask++ ;
			}
			else if (Gd_new[d] < 0.) {
				mask[d] = 192 ;
				num_mask++ ;
			}
			else {
				scale_new[d] *= 4. ;
			}
		}
		
		if (num_mask > ((double) 0.99*frames->tot_num_data))
			break ;
	}
	if (i == 5)
		fprintf(stderr, "WARNING: Could not find search bounds for %d/%d frames\n", frames->tot_num_data - num_mask, frames->tot_num_data) ;
	
	// Bounded root finding using bisection/regula falsi/Ridder's
	for (i = 0 ; i < 50 ; ++i) {
		num_mask = 0 ;
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			if (mask[d] == 255) {
				num_mask++ ;
			}
			else if (fabs(scale_old[d] - scale_new[d]) < 1.e-5) {
				mask[d] = 255 ;
				iter->scale[d] = 0.5 * (scale_old[d] + scale_new[d]) ;
			}
			else if (mask[d] == 192) {
				mask[d] = 0 ;
				//scale_latest[d] = (scale_old[d]*Gd_new[d] - scale_new[d]*Gd_old[d]) / (Gd_new[d] - Gd_old[d]) ;
				scale_latest[d] = 0.5 * (scale_old[d] + scale_new[d]) ;
				/*
				if (i % 2 == 0) {
					scale_latest[d] = 0.5 * (scale_old[d] + scale_new[d]) ;
				}
				else {
					x0 = scale_old[d] ;
					x1 = scale_latest[d] ;
					f0 = Gd_old[d] ;
					f1 = Gd_latest[d] ;
					f2 = Gd_new[d] ;
					scale_latest[d] = x1 + (x1-x0)*copysign(1, f0)*f1/sqrt(f1*f1 - f0*f2) ;
				}
				*/
			}
			else {
				//scale_latest[d] = (scale_old[d]*Gd_new[d] - scale_new[d]*Gd_old[d]) / (Gd_new[d] - Gd_old[d]) ;
				scale_latest[d] = 0.5 * (scale_old[d] + scale_new[d]) ;
				/*
				if (i % 2 == 0) {
					scale_latest[d] = 0.5 * (scale_old[d] + scale_new[d]) ;
				}
				else {
					x0 = scale_old[d] ;
					x1 = scale_latest[d] ;
					f0 = Gd_old[d] ;
					f1 = Gd_latest[d] ;
					f2 = Gd_new[d] ;
					scale_latest[d] = x1 + (x1-x0)*copysign(1, f0)*f1/sqrt(f1*f1 - f0*f2) ;
				}
				*/
			}
		}
		
		gradient_d(common, mask, scale_latest, Gd_latest) ;
		
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (mask[d] < 255) {
			if (fabs(Gd_latest[d]) < 1.e-3) {
				iter->scale[d] = scale_latest[d] ;
				mask[d] = 255 ;
				num_mask++ ;
			}
			//else if (i % 2 == 1 && Gd_latest[d] * Gd_old[d] > 0) {
			else if (Gd_latest[d] * Gd_old[d] > 0) {
				scale_old[d] = scale_latest[d] ;
				Gd_old[d] = Gd_latest[d] ;
			}
			//else if (i % 2 == 1) {
			else {
				scale_new[d] = scale_latest[d] ;
				Gd_new[d] = Gd_latest[d] ;
			}
		}

		if (num_mask == frames->tot_num_data)
			break ;
	}
	if (i == 50)
		fprintf(stderr, "WARNING: scale optimization did not converge for %d/%d frames\n", frames->tot_num_data-num_mask, frames->tot_num_data) ;
	
	MPI_Bcast(iter->scale, frames->tot_num_data, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
	
	// Free memory
	free(scale_old) ; free(scale_new) ; free(scale_latest) ;
	free(Gd_old) ; free(Gd_new) ; free(Gd_latest) ;
	free(mask) ;
	char tag[128] ;
	sprintf(tag, "(%d iterations)", i) ;
	print_max_time("scale", tag, !param->rank) ;
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

