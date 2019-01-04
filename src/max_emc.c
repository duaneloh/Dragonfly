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

#define PDIFF_THRESH 14.
#define MAX_EXP_START -1.e100

static struct timeval tm1, tm2 ;

static void allocate_memory(struct max_data*) ;
static void calculate_rescale(struct max_data*) ;
static void calculate_prob(int, struct max_data*, struct max_data*) ;
static void normalize_prob(struct max_data*, struct max_data*) ;
static double calc_psum_r(int, struct max_data*, struct max_data*) ;
static void update_tomogram(int, struct max_data*, struct max_data*) ;
static void optimize_tomogram(int, struct max_data*, struct max_data*) ;
static void merge_tomogram(int, struct max_data*) ;
static void combine_information_omp(struct max_data*, struct max_data*) ;
static double combine_information_mpi(struct max_data*) ;
static void free_memory(struct max_data*) ;
static void print_max_time(char*, char*, int) ;

double maximize() {
	double avg_likelihood ;
	gettimeofday(&tm1, NULL) ;
	iter->mutual_info = 0. ;
	struct max_data *common_data = malloc(sizeof(struct max_data)) ;
	common_data->refinement = 0 ;
	common_data->within_openmp = 0 ;
	
	allocate_memory(common_data) ;
	calculate_rescale(common_data) ;

	#pragma omp parallel default(shared)
	{
		int r ;
		struct max_data *priv_data = malloc(sizeof(struct max_data)) ;
		
		priv_data->refinement = 0 ;
		priv_data->within_openmp = 1 ;
		allocate_memory(priv_data) ;
		
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < quat->num_rot_p ; ++r)
			calculate_prob(r, priv_data, common_data) ;
		
		normalize_prob(priv_data, common_data) ;
		
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < quat->num_rot_p ; ++r) {
			if (det[0].with_bg && param->need_scaling)
				optimize_tomogram(r, priv_data, common_data) ;
			else
				update_tomogram(r, priv_data, common_data) ;
			merge_tomogram(r, priv_data) ;
		}
		
		// Combine information from different OpenMP ranks
		// This function (and the associated private arrays) will be unnecessary with
		// OpenMP 4.5 support available in GCC 6.1+ or ICC 17.0+
		combine_information_omp(priv_data, common_data) ;
		
		free_memory(priv_data) ;
	}

	avg_likelihood = combine_information_mpi(common_data) ;
	if (!param->rank)
		save_metrics(common_data) ;
	free_memory(common_data) ;
	
	return avg_likelihood ;
}

/*
static void refine_frame(int, struct dataset*, struct max_data*, struct max_data*) ;
double refine_maximize() {
	double avg_likelihood ;
	struct max_data *common_data = malloc(sizeof(struct max_data)) ;
	
	gettimeofday(&tm1, NULL) ;
	iter->mutual_info = 0. ;
	common_data->refinement = 1 ;
	common_data->within_openmp = 0 ;
	allocate_memory(common_data) ;
	
	iter->rescale = calculate_rescale(common_data) ;
	
	#pragma omp parallel default(shared)
	{
		int d ;
		struct dataset *curr = frames ;
		struct max_data *priv_data = malloc(sizeof(struct max_data)) ;
		
		priv_data->refinement = 1 ;
		priv_data->within_openmp = 1 ;
		allocate_memory(priv_data) ;
		
		while (curr != NULL) {
			#pragma omp for schedule(guided)
			for (d = 0 ; d < curr->num_data ; ++d) {
				if (frames->blacklist[curr->num_data_prev+d])
					continue ;
				refine_frame(d, curr, priv_data, common_data) ;
			}
			curr = curr->next ;
		}
		
		combine_information_omp(priv_data, common_data) ;
		free_memory(priv_data) ;
	}
	
	avg_likelihood = combine_information_mpi(common_data) ;
	if (!param->rank)
		save_metrics(common_data) ;
	free_memory(common_data) ;
	
	return avg_likelihood ;
}

void refine_frame(int d, struct dataset *curr, struct max_data *priv, struct max_data *common) {
	int r, t, pixel ;
	double max_exp = MAX_EXP_START ;
	double p_norm = 0. ;
	double new_scale = 0. ;
	double *prob = priv->probab[0] ;
	
	for (r = 0 ; r < num_rot_sub[d] ; ++r) {
		(*slice_gen)(&quat->quat[quat_sub[d][r]*5], 1., view, iter->model1, iter->size, det) ;
		
		prob[r] = common->u[quat_sub[d][r]] * iter->scale[curr->num_data_prev+d] ;
		
		for (t = 0 ; t < curr->ones[d] ; ++t) {
			pixel = curr->place_ones[curr->ones_accum[d] + t] ;
			if (det->mask[pixel] < 1)
				prob[r] += view[pixel] ;
		}
		
		for (t = 0 ; t < curr->multi[d] ; ++t) {
			pixel = curr->place_multi[curr->multi_accum[d] + t] ;
			if (det->mask[pixel] < 1)
				prob[r] += curr->count_multi[curr->multi_accum[d] + t] * view[pixel] ;
		}
		
		if (prob[r] > max_exp) {
			max_exp = prob[r] ;
			priv->rmax[curr->num_data_prev+d] = r ;
		}
	}
	
	for (r = 0 ; r < num_rot_sub[d] ; ++r) {
		prob[r] = exp(param->beta * (prob[r] - max_exp)) ;
		p_norm += prob[r] ;
	}
	
	for (r = 0 ; r < num_rot_sub[d] ; ++r) {
		prob[r] /= p_norm ;
		new_scale -= prob[r] * common->u[quat_sub[d][r]] ;
		if (prob[r] < PROB_MIN)
			continue ;
		
		memset(view, 0, det->num_pix * sizeof(double)) ;
		
		for (t = 0 ; t < curr->ones[d] ; ++t) {
			pixel = curr->place_ones[curr->ones_accum[d] + t] ;
			if (det->mask[pixel] < 2)
				view[pixel] += prob[r] / iter->scale[curr->num_data_prev+d] ;
		}
		
		for (t = 0 ; t < curr->multi[d] ; ++t) {
			pixel = curr->place_multi[curr->multi_accum[d] + t] ;
			if (det->mask[pixel] < 2)
				view[pixel] += curr->count_multi[curr->multi_accum[d] + t] * prob[r] / iter->scale[curr->num_data_prev+d] ;
		}
		
		(*slice_merge)(&quat->quat[quat_sub[d][r]*5], view, priv->model, priv->weight, iter->size, det) ;
	}
	
	iter->scale[curr->num_data_prev + d] = frames->count[curr->num_data_prev + d] / new_scale ;
	
	if ((curr->num_data_prev+d)%(frames->tot_num_data/10) == 0)
		fprintf(stderr, "\r\t\tFinished %d/%d frames", curr->num_data_prev+d, frames->tot_num_data) ;
}
*/

void allocate_memory(struct max_data *data) {
	int detn, d ;
	
	data->rmax = calloc(frames->tot_num_data, sizeof(int)) ;
	data->info = calloc(frames->tot_num_data, sizeof(double)) ;
	data->likelihood = calloc(frames->tot_num_data, sizeof(double)) ;
	data->max_exp_p = malloc(frames->tot_num_data * sizeof(double)) ;
	for (d = 0 ; d < frames->tot_num_data ; ++d)
		data->max_exp_p[d] = MAX_EXP_START ;
	if (param->modes > 1)
		data->quat_norm = calloc(param->modes * frames->tot_num_data, sizeof(double)) ;
	
	if (!data->within_openmp) { // common_data
		data->u = malloc(det[0].num_det * sizeof(double*)) ;
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
			data->u[detn] = calloc(quat->num_rot_p, sizeof(double)) ;
		data->max_exp = calloc(frames->tot_num_data, sizeof(double)) ;
		data->p_norm = calloc(frames->tot_num_data, sizeof(double)) ;
		
		memset(iter->model2, 0, param->modes*iter->vol*sizeof(double)) ;
		memset(iter->inter_weight, 0, param->modes*iter->vol*sizeof(double)) ;
		print_max_time("alloc", "", param->rank == 0) ;
	}
	else { // priv_data
		data->all_views = malloc(det[0].num_det * sizeof(double*)) ;
		data->mask = malloc(det[0].num_det * sizeof(uint8_t*)) ;
		for (d = 0 ; d < det[0].num_det ; ++d) {
			data->all_views[d] = malloc(det[d].num_pix * sizeof(double)) ;
			data->mask[d] = calloc(det[d].num_pix, sizeof(uint8_t*)) ;
		}
		
		data->model = calloc(param->modes*iter->vol, sizeof(double)) ;
		data->weight = calloc(param->modes*iter->vol, sizeof(double)) ;
		
		if (param->need_scaling && param->update_scale)
			data->psum_d = calloc(frames->tot_num_data, sizeof(double)) ;
		data->psum_r = calloc(det[0].num_det, sizeof(double)) ;
		
		data->prob = malloc(frames->tot_num_data * sizeof(double*)) ;
		data->probpos = malloc(frames->tot_num_data * sizeof(int*)) ;
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			data->prob[d] = malloc(4 * sizeof(double)) ;
			data->probpos[d] = malloc(4 * sizeof(int)) ;
		}
		data->num_prob = calloc(frames->tot_num_data, sizeof(int)) ;
		
		if (det[0].with_bg && param->need_scaling) {
			data->G_old = malloc(det[0].num_det * sizeof(double*)) ;
			data->G_new = malloc(det[0].num_det * sizeof(double*)) ;
			data->G_latest = malloc(det[0].num_det * sizeof(double*)) ;
			data->W_old = malloc(det[0].num_det * sizeof(double*)) ;
			data->W_new = malloc(det[0].num_det * sizeof(double*)) ;
			data->W_latest = malloc(det[0].num_det * sizeof(double*)) ;
			for (detn = 0 ; detn < det[0].num_det ; ++detn) {
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
	// Only calculating based on first detector and dataset
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
	
	#pragma omp parallel default(shared)
	{
		int detn, r, rotind ;
		
		#pragma omp for schedule(static,1) 
		for (r = 0 ; r < quat->num_rot_p ; ++r) {
			rotind = (r*param->num_proc + param->rank) / param->modes ;
			for (detn = 0 ; detn < det[0].num_det ; ++detn)
				data->u[detn][r] = log(quat->quat[rotind*5 + 4]) - data->u[detn][r] ;
		}
	}
	
	sprintf(res_string, "(=") ;
	for (detn = 0 ; detn < det[0].num_det ; ++detn) {
		iter->rescale[detn] = iter->mean_count[detn] / total[detn] * param->modes ;
		sprintf(res_string + strlen(res_string), " %.6e", iter->rescale[detn]) ;
	}
	sprintf(res_string + strlen(res_string), ")") ;
	print_max_time("rescale", res_string, param->rank == 0) ;
}

static int resparsify(double *vals, int *pos, int num_vals, double thresh) {
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

void calculate_prob(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, t, d, curr_d, pixel, mode, rotind, detn, old_detn = -1 ;
	struct dataset *curr = frames ;
	double pval, *view ;
	int *num_prob = priv->num_prob, **probpos = priv->probpos ;
	double **prob = priv->prob ;
	
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	mode = (r*param->num_proc + param->rank) % param->modes ;
	
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
			
			if (curr->type < 2) {
				// need_scaling is for if we want to assume variable incident intensity
				if (param->need_scaling && (param->iteration > 1 || param->known_scale))
					pval = common->u[detn][r] * iter->scale[d] ;
				else
					pval = common->u[detn][r] * iter->rescale[detn] ;
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
							pval += log(view[pixel] * iter->scale[d] + det[detn].background[pixel]) ;
					}
					
					// For each pixel with count_multi photons
					for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
						pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
						if (det[detn].mask[pixel] < 1)
							pval += curr->count_multi[curr->multi_accum[curr_d] + t] * log(view[pixel] * iter->scale[d] + det[detn].background[pixel]) ;
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
			if (pval + PDIFF_THRESH > priv->max_exp_p[d]) {
				prob[d][num_prob[d]] = pval ;
				probpos[d][num_prob[d]] = r ;
				num_prob[d]++ ;
				
				// If num_prob is a power of two, expand array
				if (num_prob[d] >= 4 && (num_prob[d] & (num_prob[d] - 1)) == 0) {
					prob[d] = realloc(prob[d], num_prob[d] * 2 * sizeof(double)) ;
					probpos[d] = realloc(probpos[d], num_prob[d] * 2 * sizeof(int)) ;
				}
			}
			
			// Note maximum log-likelihood for each frame among 'r's tested by this MPI rank and OMP rank
			// Recalculate sparse array with new maximum
			if (pval > priv->max_exp_p[d]) {
				priv->max_exp_p[d] = pval ;
				priv->rmax[d] = r*param->num_proc + param->rank ;
				num_prob[d] = resparsify(prob[d], probpos[d], num_prob[d], pval - PDIFF_THRESH) ;
			}
		}
		
		curr = curr->next ;
		dset++ ;
	}
	
	if ((r*param->num_proc + param->rank)%(quat->num_rot * param->modes / 10) == 0)
		fprintf(stderr, "\t\tFinished r = %d/%d\n", r*param->num_proc + param->rank, quat->num_rot * param->modes) ;
	if (r == quat->num_rot_p - 1)
		print_max_time("prob", "", param->rank == 0) ;
}

void normalize_prob(struct max_data *priv, struct max_data *common) {
	int r, d, omp_rank = omp_get_thread_num() ;
	double *priv_norm = calloc(frames->tot_num_data, sizeof(double)) ;
	
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
		priv_norm[d] += exp(param->beta * (priv->prob[d][r] - common->max_exp[d])) ;
	else
		priv_norm[d] += exp(param->beta * (priv->prob[d][r] - common->max_exp[d]) / 2. / param->sigmasq) ;
	
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

double calc_psum_r(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, d, curr_d, detn, rotind, mode, t, ind ;
	double temp, scalemin ;
	struct dataset *curr = frames ;
	double **prob = priv->prob ;
	int **probpos = priv->probpos, *num_prob = priv->num_prob ;
	
	scalemin = DBL_MAX ;
	memset(priv->psum_r, 0, (det[0].num_det)*sizeof(double)) ;
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	mode = (r*param->num_proc + param->rank) % param->modes ;
	
	while (curr != NULL) {
		detn = det[0].mapping[dset] ;
		for (curr_d = 0 ; curr_d < curr->num_data ; ++curr_d) {
			// Calculate frame number in full list
			d = curr->num_data_prev + curr_d ;

			// check if frame is blacklisted
			if (frames->blacklist[d])
				continue ;
			
			// check if current frame has significant probability
			ind = -1 ;
			for (t = 0 ; t < num_prob[d] ; ++t)
			if (r == probpos[d][t]) {
				ind = t ;
				break ;
			}
			if (ind == -1)
				continue ;
			
			// Exponentiate log-likelihood and normalize to get probabilities
			temp = prob[d][ind] ;
			if (frames->type < 2)
				prob[d][ind] = exp(param->beta*(prob[d][ind] - common->max_exp[d])) / common->p_norm[d] ; 
			else
				prob[d][ind] = exp(param->beta*(prob[d][ind] - common->max_exp[d]) / 2. / param->sigmasq) / common->p_norm[d] ;
			
			//if (param->need_scaling)
			//	priv->likelihood[d] += prob[d][ind] * (temp - frames->sum_fact[d] + frames->count[d]*log(iter->scale[d])) ;
			//else
			priv->likelihood[d] += prob[d][ind] * (temp - frames->sum_fact[d]) ;
			
			// Calculate denominator for update rule
			if (param->need_scaling) {
				priv->psum_r[detn] += prob[d][ind] * iter->scale[d] ;
				
				// Calculate denominator for scale factor update rule
				if (param->update_scale)
					priv->psum_d[d] -= prob[d][ind] * common->u[detn][r] * iter->rescale[detn] ;
			}
			else
				priv->psum_r[detn] += prob[d][ind] ; 
			
			// Skip if probability is very low (saves time)
			if (prob[d][ind] < PROB_MIN)
				continue ;
			
			// If multiple modes, calculate occupancy of frame into each mode
			if (param->modes > 1)
				priv->quat_norm[d*param->modes + mode] += prob[d][ind] ;
			
			// Calculate mutual information of probability distribution
			priv->info[d] += prob[d][ind] * log(prob[d][ind] / quat->quat[rotind*5 + 4] * param->modes) ;
			
			if (param->need_scaling && iter->scale[d] < scalemin)
				scalemin = iter->scale[d] ;
		}
		dset++ ;
		curr = curr->next ;
	}
	
	return scalemin ;
}

void update_tomogram(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, t, d, curr_d, pixel, detn, ind ;
	struct dataset *curr ;
	double *view ;
	double **prob = priv->prob ;
	int **probpos = priv->probpos, *num_prob = priv->num_prob ;
	
	if (merge_frames != NULL) {
		if (!param->rank && !r)
			fprintf(stderr, "Merging with different data file: %s\n", merge_frames->filename) ;
		curr = merge_frames ;
	}
	else
		curr = frames ;
	for (detn = 0 ; detn < det[0].num_det ; ++detn) 
		memset(priv->all_views[detn], 0, det[detn].num_pix*sizeof(double)) ;
	
	calc_psum_r(r, priv, common) ;
	
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
			
			// check if current frame has significant probability
			ind = -1 ;
			for (t = 0 ; t < num_prob[d] ; ++t)
			if (r == probpos[d][t]) {
				ind = t ;
				break ;
			}
			if (ind == -1)
				continue ;
			
			
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

static void gradient_rt(int r, struct max_data *priv, struct max_data *common, double **views, double **gradients) {
	int dset = 0, t, d, curr_d, pixel, detn ;
	double val, *grad, *view ;
	struct dataset *curr = frames ;
	
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
			else {
				grad[t] = 0. ;
			}
		}
	}
	
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
			
			// Skip if probability is very low (saves time)
			if (!(common->prob[d][r] > PROB_MIN))
				continue ;
			
			// Currently only working with type-0 data
			// For each pixel with one photon
			for (t = 0 ; t < curr->ones[curr_d] ; ++t) {
				pixel = curr->place_ones[curr->ones_accum[curr_d] + t] ;
				if (priv->mask[detn][pixel] < 128) {
					val = view[pixel] * iter->scale[d] + det[detn].background[pixel] ;
					grad[pixel] += common->prob[d][r] * iter->scale[d] / val ;
					priv->mask[detn][pixel] = 0 ;
				}
				else if (priv->mask[detn][pixel] == 128) {
					grad[pixel] += common->prob[d][r] ;
				}
			}
			
			// For each pixel with count_multi photons
			for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
				pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
				if (priv->mask[detn][pixel] < 128) {
					val = view[pixel] * iter->scale[d] + det[detn].background[pixel] ;
					grad[pixel] += common->prob[d][r] * curr->count_multi[curr->multi_accum[curr_d] + t] * iter->scale[d] / val ;
					priv->mask[detn][pixel] = 0 ;
				}
				else if (priv->mask[detn][pixel] == 128) {
					grad[pixel] += common->prob[d][r] * curr->count_multi[curr->multi_accum[curr_d] + t] ;
				}
			}
		}
		
		dset++ ;
		curr = curr->next ;
	}
}

void optimize_tomogram(int r, struct max_data *priv, struct max_data *common) {
	double scalemin ;
	
	// Calculate pixel-independent part of likelihood: \sum_d (P_dr * phi_d)
	// Also calculate and return (phi_d_max if P_dr > PROB_MIN)
	scalemin = calc_psum_r(r, priv, common) ;
	
	/*
	==========================
	Root finding on derivative
	==========================
	* 1. Calculate gradient of likelihood G_old at W_old = 0
	* 2. Calculate G_new at W_new = <large> or -B_t/phi_d_min depending on sign of G_old
	* 3. Iterate till convergence:
	* 	If (Regula falsi): W_latest = (W_old*G_new - W_new*G_old) / (G_new - G_old)
	* 	If (Bisection): W_latest = 0.5 * (W_old + W_new)
	* 	Calculate G_latest at W_latest
	* 	If G_latest*G_old > 0:
	* 		W_old = W_latest ; G_old = G_latest
	* 	Else:
	* 		W_new = W_latest ; G_new = G_latest
	*/
	int i, t, detn ;
	int nmask, tot_num_pix = 0 ;
	double val ;
	
	nmask = 0 ;
	for (detn = 0 ; detn < det[0].num_det ; ++detn) {
		memset(priv->all_views[detn], 0, det[detn].num_pix*sizeof(double)) ;
		memset(priv->G_old[detn], 0, det[detn].num_pix*sizeof(double)) ;
		memset(priv->G_new[detn], 0, det[detn].num_pix*sizeof(double)) ;
		memset(priv->G_latest[detn], 0, det[detn].num_pix*sizeof(double)) ;
		memset(priv->W_old[detn], 0, det[detn].num_pix*sizeof(double)) ;
		memset(priv->W_new[detn], 0, det[detn].num_pix*sizeof(double)) ;
		memset(priv->W_latest[detn], 0, det[detn].num_pix*sizeof(double)) ;
		tot_num_pix += det[detn].num_pix ;
		
		for (t = 0 ; t < det[detn].num_pix ; ++t) {
			if (det[detn].mask[t] < 2) {
				if (det[detn].background[t] / scalemin < 1.e-2 * det[detn].powder[t]) {
					priv->mask[detn][t] = 128 ;
				}
				else {
					priv->W_new[detn][t] = 1.e-8 - det[detn].background[t] / scalemin ;
					priv->mask[detn][t] = 0 ;
				}
			}
			else {
				priv->mask[detn][t] = 255 ;
			}
		}
	}
	
	gradient_rt(r, priv, common, priv->W_old, priv->G_old) ;
	for (detn = 0 ; detn < det[0].num_det ; ++detn)
	for (t = 0 ; t < det[detn].num_pix ; ++t) {
		if (fabs(priv->G_old[detn][t]) < 1.e-6) {
			priv->all_views[detn][t] = 0. ;
			priv->mask[detn][t] = 255 ;
		}
		else if (priv->G_old[detn][t] > 0.) {
			val = frames->tot_mean_count / det[detn].num_pix ;
			priv->W_new[detn][t] = det[detn].powder[t] > val ? det[detn].powder[t] : val ; // Large number
			priv->W_new[detn][t] *= 1000. ;
		}
	}
	
	for (i = 0 ; i < 5 ; ++i) {
		nmask = 0 ;
		gradient_rt(r, priv, common, priv->W_new, priv->G_new) ;
		
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
		for (t = 0 ; t < det[detn].num_pix ; ++t) {
			if (priv->mask[detn][t] == 0) {
				/*
				if (fabs(priv->G_new[detn][t] < 1.e-6)) {
					priv->all_views[detn][t] = priv->W_new[detn][t] ;
					priv->mask[detn][t] = 255 ;
					nmask++ ;
				}
				*/
				if (priv->G_old[detn][t] < 0. && priv->G_new[detn][t] < 0.) {
					priv->W_new[detn][t] = 1.e-8*pow(0.01, i+1) - det[detn].background[t] / scalemin ;
				}
				else if (priv->G_old[detn][t] > 0. && priv->G_new[detn][t] > 0.) {
					priv->W_new[detn][t] *= 100 ;
				}
				else {
					priv->mask[detn][t] = 192 ;
					nmask++ ;
				}
			}
			else {
				nmask++ ;
			}
		}
		
		if (nmask == tot_num_pix)
			break ;
	}
	
	for (i = 0 ; i < 50 ; ++i) { // Doing 50 iterations
		nmask = 0 ;
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
		for (t = 0 ; t < det[detn].num_pix ; ++t) {
			if (priv->mask[detn][t] == 255) { // Already optimal
				nmask++ ;
			}
			else if (priv->mask[detn][t] == 1) { // No photon at pixel
				priv->mask[detn][t] = 255 ;
				priv->all_views[detn][t] = 0. ;
				nmask++ ;
			}
			else if (priv->mask[detn][t] == 128) { // Negligible background
				priv->all_views[detn][t] = priv->G_old[detn][t] / priv->psum_r[detn] ;
				priv->mask[detn][t] = 255 ;
				nmask++ ;
			}
			else if (priv->mask[detn][t] == 192) { // W_new had to be adjusted
				priv->mask[detn][t] = 0 ;
				priv->W_latest[detn][t] = 0.5 * (priv->W_old[detn][t] + priv->W_new[detn][t]) ;
			}
			else if (priv->mask[detn][t] == 0) {
				// Regula falsi (secant) update
				//priv->W_latest[detn][t] = (priv->W_old[detn][t]*priv->G_new[detn][t] - priv->W_new[detn][t]*priv->G_old[detn][t]) / (priv->G_new[detn][t] - priv->G_old[detn][t]) ;
				// Bisection update
				priv->W_latest[detn][t] = 0.5 * (priv->W_old[detn][t] + priv->W_new[detn][t]) ;
			}
		}
		
		gradient_rt(r, priv, common, priv->W_latest, priv->G_latest) ;
		
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
		for (t = 0 ; t < det[detn].num_pix ; ++t)
		if (priv->mask[detn][t] < 255) {
			if (fabs(priv->G_latest[detn][t]) < 1.e-5) {
				priv->all_views[detn][t] = priv->W_latest[detn][t] ;
				priv->mask[detn][t] = 255 ;
				nmask++ ;
			}
			else if (priv->G_latest[detn][t] * priv->G_old[detn][t] > 0) {
				priv->W_old[detn][t] = priv->W_latest[detn][t] ;
				priv->G_old[detn][t] = priv->G_latest[detn][t] ;
			}
			else {
				priv->W_new[detn][t] = priv->W_latest[detn][t] ;
				priv->G_new[detn][t] = priv->G_latest[detn][t] ;
			}
		}
		
		if (nmask == tot_num_pix)
			break ;
	}
	if (i == 50)
		fprintf(stderr, "%.4d not converged, %d\n", r, nmask) ;
}

void merge_tomogram(int r, struct max_data *priv) {
	int detn, mode, rotind ;
	
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	mode = (r*param->num_proc + param->rank) % param->modes ;
	
	// If no data frame has any probability for this orientation, don't merge
	for (detn = 0 ; detn < det[0].num_det ; ++detn)
	if (priv->psum_r[detn] > 0.)
		(*slice_merge)(&quat->quat[rotind*5], priv->all_views[detn], &priv->model[mode*iter->vol], &priv->weight[mode*iter->vol], iter->size, &det[detn]) ;
	
	if ((r*param->num_proc + param->rank)%(quat->num_rot * param->modes / 10) == 0)
		fprintf(stderr, "\t\tFinished r = %d/%d\n", r*param->num_proc + param->rank, quat->num_rot * param->modes) ;
}

void combine_information_omp(struct max_data *priv, struct max_data *common) {
	int d, omp_rank = omp_get_thread_num() ;
	long x ;
	
	#pragma omp critical(model)
	{
		for (x = 0 ; x < param->modes * iter->vol ; ++x) {
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
		if (omp_rank == 0)
			memset(iter->scale, 0, frames->tot_num_data * sizeof(double)) ;
		#pragma omp barrier
		
		#pragma omp critical(scale)
		{
			for (d = 0 ; d < frames->tot_num_data ; ++d)
			if (!frames->blacklist[d])
				iter->scale[d] += priv->psum_d[d] ;
		}
	}
	
	if (param->modes > 1) {
		#pragma omp critical(quat_norm)
		for (d = 0 ; d < frames->tot_num_data * param->modes ; ++d)
			common->quat_norm[d] += priv->quat_norm[d] ;
	}
	print_max_time("update", "", param->rank == 0 && omp_rank == 0) ;
}

double combine_information_mpi(struct max_data *data) {
	int d ;
	double avg_likelihood = 0. ;
	
	// Combine 3D volumes from all MPI ranks
	if (param->rank) {
		MPI_Reduce(iter->model2, iter->model2, param->modes*iter->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
		MPI_Reduce(iter->inter_weight, iter->inter_weight, param->modes*iter->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
	}
	else {
		MPI_Reduce(MPI_IN_PLACE, iter->model2, param->modes*iter->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
		MPI_Reduce(MPI_IN_PLACE, iter->inter_weight, param->modes*iter->vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
	}
	
	// Combine mutual info and likelihood from all MPI ranks
	MPI_Allreduce(MPI_IN_PLACE, data->likelihood, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	MPI_Allreduce(MPI_IN_PLACE, data->info, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	if (param->modes > 1)
		MPI_Allreduce(MPI_IN_PLACE, data->quat_norm, frames->tot_num_data*param->modes, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (!frames->blacklist[d]) {
		iter->mutual_info += data->info[d] ;
		avg_likelihood += data->likelihood[d] ;
	}
	
	iter->mutual_info /= (frames->tot_num_data - frames->num_blacklist) ;
	avg_likelihood /= (frames->tot_num_data - frames->num_blacklist) ;
	
	// Calculate updated scale factor using count[d] (total photons in frame d)
	if (param->need_scaling && param->update_scale) {
		// Combine scale factor information from all MPI ranks
		MPI_Allreduce(MPI_IN_PLACE, iter->scale, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
		
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (!frames->blacklist[d])
			iter->scale[d] = frames->count[d] / iter->scale[d] ;
	}
	
	return avg_likelihood ;
}

void free_memory(struct max_data *data) {
	free(data->max_exp_p) ;
	free(data->info) ;
	free(data->likelihood) ;
	free(data->rmax) ;
	if (param->modes > 1)
		free(data->quat_norm) ;
	
	if (!data->within_openmp) {
		free(data->max_exp) ;
		free(data->u) ;
		free(data->p_norm) ;
	}
	else {
		int detn, d ;
		for (d = 0 ; d < det[0].num_det ; ++d) {
			free(data->all_views[d]) ;
			free(data->mask[d]) ;
		}
		free(data->all_views) ;
		free(data->mask) ;
		if (param->need_scaling && param->update_scale)
			free(data->psum_d) ;
		free(data->model) ;
		free(data->weight) ;
		free(data->psum_r) ;
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			free(data->prob[d]) ;
			free(data->probpos[d]) ;
		}
		free(data->prob) ;
		free(data->probpos) ;
		free(data->num_prob) ;
		if (det[0].with_bg && param->need_scaling) {
			for (detn = 0 ; detn < det[0].num_det ; ++detn) {
				free(data->G_old[detn]) ;
				free(data->G_new[detn]) ;
				free(data->G_latest[detn]) ;
				free(data->W_old[detn]) ;
				free(data->W_new[detn]) ;
				free(data->W_latest[detn]) ;
			}
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

