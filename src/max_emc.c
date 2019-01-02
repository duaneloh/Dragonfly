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

struct max_data {
	int refinement, within_openmp ;
	
	// Common only
	double *max_exp, *u, **probab ;
	
	// Private only
	double *model, *weight, *scale ;
	double **all_views ;
	
	// Both
	double *max_exp_p, *p_sum ;
	double *info, *likelihood ;
	int *rmax ;
	double *quat_norm ;
} ;
static struct timeval tm1, tm2 ;

static void allocate_memory(struct max_data*) ;
static double calculate_rescale(struct max_data*) ;
static void calculate_prob(int, struct max_data*, struct max_data*) ;
static void normalize_prob(struct max_data*, struct max_data*) ;
static void update_tomogram(int, struct max_data*, struct max_data*) ;
static void merge_tomogram(int, struct max_data*) ;
//static void refine_frame(int, struct dataset*, struct max_data*, struct max_data*) ;
static void combine_information_omp(struct max_data*, struct max_data*) ;
static double combine_information_mpi(struct max_data*) ;
static void save_output(struct max_data*) ;
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
	iter->rescale = calculate_rescale(common_data) ;

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
			update_tomogram(r, priv_data, common_data) ;
			merge_tomogram(r, priv_data) ;
			
			free(common_data->probab[r]) ;
		}
		
		// Combine information from different OpenMP ranks
		// This function (and the associated private arrays) will be unnecessary with
		// OpenMP 4.5 support available in GCC 6.1+ or ICC 17.0+
		combine_information_omp(priv_data, common_data) ;
		
		free_memory(priv_data) ;
	}

	avg_likelihood = combine_information_mpi(common_data) ;
	if (!param->rank)
		save_output(common_data) ;
	free_memory(common_data) ;
	
	return avg_likelihood ;
}

/*
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
		save_output(common_data) ;
	free_memory(common_data) ;
	
	return avg_likelihood ;
}

void refine_frame(int d, struct dataset *curr, struct max_data *priv, struct max_data *common) {
	int r, t, pixel ;
	double max_exp = -DBL_MAX ;
	double p_sum = 0. ;
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
		p_sum += prob[r] ;
	}
	
	for (r = 0 ; r < num_rot_sub[d] ; ++r) {
		prob[r] /= p_sum ;
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
	int d, r ;
	
	data->rmax = calloc(frames->tot_num_data, sizeof(int)) ;
	data->info = calloc(frames->tot_num_data, sizeof(double)) ;
	data->likelihood = calloc(frames->tot_num_data, sizeof(double)) ;
	if (!data->refinement) {
		data->max_exp_p = malloc(frames->tot_num_data * sizeof(double)) ;
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			data->max_exp_p[d] = -DBL_MAX ;
	}
	if (param->modes > 1)
		data->quat_norm = calloc(param->modes, sizeof(double)) ;
	
	if (!data->within_openmp) { // common_data
		data->u = calloc(quat->num_rot_p, sizeof(double)) ;
		if (!data->refinement) { // Global search
			data->probab = malloc(quat->num_rot_p * sizeof(double*)) ;
			data->max_exp = calloc(frames->tot_num_data, sizeof(double)) ;
			data->p_sum = calloc(frames->tot_num_data, sizeof(double)) ;
			for (r = 0 ; r < quat->num_rot_p ; ++r)
				data->probab[r] = malloc(frames->tot_num_data * sizeof(double)) ;
		}
		else {
			data->probab = malloc(1 * sizeof(double*)) ;
			//data->probab[0] = malloc(num_rot_sub_max * sizeof(double)) ;
		}
		
		memset(iter->model2, 0, param->modes*iter->vol*sizeof(double)) ;
		memset(iter->inter_weight, 0, param->modes*iter->vol*sizeof(double)) ;
		print_max_time("alloc", "", param->rank == 0) ;
	}
	else { // priv_data
		data->all_views = malloc(det[0].num_det * sizeof(double*)) ;
		for (d = 0 ; d < det[0].num_det ; ++d)
			data->all_views[d] = malloc(det[d].num_pix * sizeof(double)) ;
		data->model = calloc(param->modes*iter->vol, sizeof(double)) ;
		data->weight = calloc(param->modes*iter->vol, sizeof(double)) ;
		if (param->need_scaling)
			data->scale = calloc(frames->tot_num_data, sizeof(double)) ;
		if (!data->refinement) // Global search
			data->p_sum = calloc(det[0].num_det, sizeof(double)) ;
	}
}

double calculate_rescale(struct max_data *data) {
	int r, t ;
	double total = 0. ;
	char res_string[1024] ;
	
	// Calculate rescale factor by calculating mean model value over detector
	// Only calculating based on first detector and dataset
	#pragma omp parallel default(shared) private(r, t)
	{
		double *view = malloc(det[0].num_pix * sizeof(double)) ;
		int mode, rotind ;
		
		#pragma omp for schedule(static,1) reduction(+:total)
		for (r = 0 ; r < quat->num_rot_p ; ++r) {
			rotind = (r*param->num_proc + param->rank) / param->modes ;
			mode = (r*param->num_proc + param->rank) % param->modes ;
			// Second argument being 0. tells slice_gen to generate un-rescaled tomograms
			(*slice_gen)(&quat->quat[rotind*5], 0., view, &iter->model1[mode*iter->vol], iter->size, det) ;
			
			for (t = 0 ; t < det[0].num_pix ; ++t)
			if (det[0].mask[t] < 1)
				data->u[r] += view[t] ;
			
			total += quat->quat[rotind*5 + 4] * data->u[r] ;
		}
		
		free(view) ;
	}
	
	MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	#pragma omp parallel default(shared) private(r)
	{
		int rotind ;
		
		#pragma omp for schedule(static,1) 
		for (r = 0 ; r < quat->num_rot_p ; ++r) {
			rotind = (r*param->num_proc + param->rank) / param->modes ;
			data->u[r] = log(quat->quat[rotind*5 + 4]) - data->u[r] ;
		}
	}
	
	sprintf(res_string, "(= %.6e)", frames[0].mean_count / total) ;
	print_max_time("rescale", res_string, param->rank == 0) ;
	
	return frames[0].mean_count / total ;
}

void calculate_prob(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, t, d, pixel, mode, rotind, detn, old_detn = -1 ;
	struct dataset *curr = frames ;
	double *view, *prob = common->probab[r] ;
	
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	mode = (r*param->num_proc + param->rank) % param->modes ;
	
	// Linked list of data sets from different files
	while (curr != NULL) {
		//Calculate slice for current detector
		detn = det[0].mapping[dset] ;
		view = priv->all_views[detn] ;
		if (detn != old_detn)
			(*slice_gen)(&quat->quat[rotind*5], 1., view, &iter->model1[mode*iter->vol], iter->size, &det[detn]) ;
		old_detn = detn ;
		
		// For each frame in data set
		for (d = 0 ; d < curr->num_data ; ++d) {
			// check if frame is blacklisted
			if (frames->blacklist[curr->num_data_prev+d])
				continue ;
			
			if (curr->type < 2) {
				// need_scaling is for if we want to assume variable incident intensity
				if (param->need_scaling && (param->iteration > 1 || param->known_scale))
					prob[curr->num_data_prev+d] = common->u[r] * iter->scale[curr->num_data_prev+d] ;
				else
					prob[curr->num_data_prev+d] = common->u[r] * iter->rescale ;
			}
			else {
				prob[curr->num_data_prev+d] = 0. ;
			}
			
			if (curr->type == 0) {
				// For each pixel with one photon
				for (t = 0 ; t < curr->ones[d] ; ++t) {
					pixel = curr->place_ones[curr->ones_accum[d] + t] ;
					if (det[detn].mask[pixel] < 1)
						prob[curr->num_data_prev+d] += view[pixel] ;
				}
				
				// For each pixel with count_multi photons
				for (t = 0 ; t < curr->multi[d] ; ++t) {
					pixel = curr->place_multi[curr->multi_accum[d] + t] ;
					if (det[detn].mask[pixel] < 1)
						prob[curr->num_data_prev+d] += curr->count_multi[curr->multi_accum[d] + t] * view[pixel] ;
				}
			}
			else if (curr->type == 1) {
				for (t = 0 ; t < det[detn].num_pix ; ++t)
				if (det[detn].mask[t] < 1)
					prob[curr->num_data_prev+d] += curr->int_frames[d*curr->num_pix + t] * view[t] ;
			}
			else if (curr->type == 2) { // Gaussian EMC for double precision data without scaling
				for (t = 0 ; t < det[detn].num_pix ; ++t)
				if (det[detn].mask[t] < 1)
					prob[curr->num_data_prev+d] -= pow(curr->frames[d*curr->num_pix + t] - view[t]*iter->rescale, 2.) ;
			}
			
			// Note maximum log-likelihood for each frame among 'r's tested by this MPI rank and OMP rank
			if (prob[curr->num_data_prev+d] > priv->max_exp_p[curr->num_data_prev+d]) {
				priv->max_exp_p[curr->num_data_prev+d] = prob[curr->num_data_prev+d] ;
				priv->rmax[curr->num_data_prev+d] = r*param->num_proc + param->rank ;
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
	double *priv_sum = calloc(frames->tot_num_data, sizeof(double)) ;
	
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
		if (common->max_exp[d] != common->max_exp_p[d] || common->max_exp_p[d] == -DBL_MAX)
			common->rmax[d] = -1 ;
		
		MPI_Allreduce(MPI_IN_PLACE, common->rmax, frames->tot_num_data, MPI_INT, MPI_MAX, MPI_COMM_WORLD) ;
	}
	#pragma omp barrier
	
	#pragma omp for schedule(static,1)
	for (r = 0 ; r < quat->num_rot_p ; ++r)
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (frames->type < 2)
		priv_sum[d] += exp(param->beta * (common->probab[r][d] - common->max_exp[d])) ;
	else
		priv_sum[d] += exp(param->beta * (common->probab[r][d] - common->max_exp[d]) / 2. / param->sigmasq) ;
	
	#pragma omp critical(psum)
	{
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			common->p_sum[d] += priv_sum[d] ;
	}
	#pragma omp barrier
	
	if (omp_rank == 0)
		MPI_Allreduce(MPI_IN_PLACE, common->p_sum, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	#pragma omp barrier
	
	free(priv_sum) ;
	print_max_time("psum", "", param->rank == 0 && omp_rank == 0) ;
}

void update_tomogram(int r, struct max_data *priv, struct max_data *common) {
	int dset = 0, t, d, pixel, detn, rotind ;
	double temp ;
	struct dataset *curr ;
	double *view, *prob = common->probab[r] ;
	
	if (merge_frames != NULL) {
		if (!param->rank && !r)
			fprintf(stderr, "Merging with different data file: %s\n", merge_frames->filename) ;
		curr = merge_frames ;
	}
	else
		curr = frames ;
	memset(priv->p_sum, 0, (det[0].num_det)*sizeof(double)) ;
	for (detn = 0 ; detn < det[0].num_det ; ++detn) 
		memset(priv->all_views[detn], 0, det[detn].num_pix*sizeof(double)) ;
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	
	while (curr != NULL) {
		//Calculate slice for current detector
		detn = det[0].mapping[dset] ;
		view = priv->all_views[detn] ;
		
		for (d = 0 ; d < curr->num_data ; ++d) {
			// check if frame is blacklisted
			if (frames->blacklist[curr->num_data_prev+d])
				continue ;
			
			// Exponentiate log-likelihood and normalize to get probabilities
			temp = prob[curr->num_data_prev+d] ;
			if (frames->type < 2)
				prob[curr->num_data_prev+d] = exp(param->beta*(prob[curr->num_data_prev+d] - common->max_exp[curr->num_data_prev+d])) / common->p_sum[curr->num_data_prev+d] ; 
			else
				prob[curr->num_data_prev+d] = exp(param->beta*(prob[curr->num_data_prev+d] - common->max_exp[curr->num_data_prev+d]) / 2. / param->sigmasq) / common->p_sum[curr->num_data_prev+d] ;
			
			//if (param->need_scaling)
			//	priv->likelihood[curr->num_data_prev+d] += prob[curr->num_data_prev+d] * (temp - frames->sum_fact[curr->num_data_prev+d] + frames->count[curr->num_data_prev+d]*log(iter->scale[curr->num_data_prev+d])) ;
			//else
			priv->likelihood[curr->num_data_prev+d] += prob[curr->num_data_prev+d] * (temp - frames->sum_fact[curr->num_data_prev+d]) ;
			
			// Calculate denominator for update rule
			if (param->need_scaling) {
				priv->p_sum[detn] += prob[curr->num_data_prev+d] * iter->scale[curr->num_data_prev+d] ;
				// Calculate denominator for scale factor update rule
				if (param->iteration > 1)
					priv->scale[curr->num_data_prev+d] -= prob[curr->num_data_prev+d] * common->u[r] ;
				else
					priv->scale[curr->num_data_prev+d] -= prob[curr->num_data_prev+d] * common->u[r] * iter->rescale ;
			}
			else
				priv->p_sum[detn] += prob[curr->num_data_prev+d] ; 
			
			// Skip if probability is very low (saves time)
			if (!(prob[curr->num_data_prev+d] > PROB_MIN))
				continue ;
			
			// Calculate mutual information of probability distribution
			priv->info[curr->num_data_prev+d] += prob[curr->num_data_prev+d] * log(prob[curr->num_data_prev+d] / quat->quat[rotind*5 + 4]) ;
			
			if (curr->type == 0) {
				// For all pixels with one photon
				for (t = 0 ; t < curr->ones[d] ; ++t) {
					pixel = curr->place_ones[curr->ones_accum[d] + t] ;
					if (det[detn].mask[pixel] < 2)
						view[pixel] += prob[curr->num_data_prev+d] ;
				}
				
				// For all pixels with count_multi photons
				for (t = 0 ; t < curr->multi[d] ; ++t) {
					pixel = curr->place_multi[curr->multi_accum[d] + t] ;
					if (det[detn].mask[pixel] < 2)
						view[pixel] += curr->count_multi[curr->multi_accum[d] + t] * prob[curr->num_data_prev+d] ;
				}
			}
			else if (curr->type == 1) {
				for (t = 0 ; t < curr->num_pix ; ++t)
					view[t] += curr->int_frames[d*curr->num_pix + t] * prob[curr->num_data_prev+d] ;
			}
			else if (curr->type == 2) { // Gaussian EMC update without scaling
				for (t = 0 ; t < curr->num_pix ; ++t)
				if (det[detn].mask[t] < 2)
					view[t] += curr->frames[d*curr->num_pix + t] * prob[curr->num_data_prev+d] ;
			}
		}
		
		curr = curr->next ;
		dset++ ;
	}
	
	if (param->modes > 1) {
		for (detn = 0 ; detn < det[0].num_det ; ++detn)
			priv->quat_norm[(r*param->num_proc + param->rank) % param->modes] += priv->p_sum[detn] ;
	}
}

void merge_tomogram(int r, struct max_data *priv) {
	int detn, t, mode, rotind ;
	double *view ;
	
	rotind = (r*param->num_proc + param->rank) / param->modes ;
	mode = (r*param->num_proc + param->rank) % param->modes ;
	
	// If no data frame has any probability for this orientation, don't merge
	// Otherwise divide the updated tomogram by the sum over all probabilities and merge
	for (detn = 0 ; detn < det[0].num_det ; ++detn) {
		view = priv->all_views[detn] ;
		if (priv->p_sum[detn] > 0.) {
			for (t = 0 ; t < det[detn].num_pix ; ++t)
				view[t] /= priv->p_sum[detn] ;
			
			(*slice_merge)(&quat->quat[rotind*5], view, &priv->model[mode*iter->vol], &priv->weight[mode*iter->vol], iter->size, &det[detn]) ;
		}
	}
	
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
	
	if (param->need_scaling) {
		if (omp_rank == 0)
			memset(iter->scale, 0, frames->tot_num_data * sizeof(double)) ;
		#pragma omp barrier
		
		#pragma omp critical(scale)
		{
			for (d = 0 ; d < frames->tot_num_data ; ++d)
			if (!frames->blacklist[d])
				iter->scale[d] += priv->scale[d] ;
		}
	}
	
	if (param->modes > 1) {
		#pragma omp critical(quat_norm)
		for (d = 0 ; d < param->modes ; ++d)
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
		MPI_Allreduce(MPI_IN_PLACE, data->quat_norm, param->modes, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (!frames->blacklist[d]) {
		iter->mutual_info += data->info[d] ;
		avg_likelihood += data->likelihood[d] ;
	}
	
	// Calculate updated scale factor using count[d] (total photons in frame d)
	if (param->need_scaling) {
		// Combine scale factor information from all MPI ranks
		MPI_Allreduce(MPI_IN_PLACE, iter->scale, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
		
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (!frames->blacklist[d])
			iter->scale[d] = frames->count[d] / iter->scale[d] ;
	}
	
	iter->mutual_info /= (frames->tot_num_data - frames->num_blacklist) ;
	avg_likelihood /= (frames->tot_num_data - frames->num_blacklist) ;
	
	return avg_likelihood ;
}

void save_output(struct max_data *data) {
	int d, r ;
	
	if (param->need_scaling) {	
		char fname[100] ;
		sprintf(fname, "%s/scale/scale_%.3d.dat", param->output_folder, param->iteration) ;
		FILE *fp_scale = fopen(fname, "w") ;
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			fprintf(fp_scale, "%.15e\n", iter->scale[d]) ;
		fclose(fp_scale) ;
	}
	
	if (param->modes > 1 && param->rank == 0) {
		fprintf(stderr, "Mode occupancies: ") ;
		for (r = 0 ; r < param->modes ; ++r)
			fprintf(stderr, "%.3f ", data->quat_norm[r]/(frames->tot_num_data - frames->num_blacklist)) ;
		fprintf(stderr, "\n") ;
	}
	//if (param->modes > 1)
	//for (r = 0 ; r < quat->num_rot ; ++r)
	//	quat->quat[r*5 + 4] = data->quat_norm[r/param->rot_per_mode] / (frames->tot_num_data - frames->num_blacklist) ;
	
	// Print frame-by-frame mutual information, likelihood, and most likely orientations to file
	char fname[1024] ;
	sprintf(fname, "%s/mutualInfo/info_%.3d.dat", param->output_folder, param->iteration) ;
	FILE *fp_info = fopen(fname, "w") ;
	sprintf(fname, "%s/likelihood/likelihood_%.3d.dat", param->output_folder, param->iteration) ;
	FILE *fp_likelihood = fopen(fname, "w") ;
	sprintf(fname, "%s/orientations/orientations_%.3d.bin", param->output_folder, param->iteration) ;
	FILE *fp_rmax = fopen(fname, "w") ;
	
	fwrite(data->rmax, sizeof(int), frames->tot_num_data, fp_rmax) ;
	for (d = 0 ; d < frames->tot_num_data ; ++d) {
		fprintf(fp_info, "%.6e\n", data->info[d]) ;
		fprintf(fp_likelihood, "%.6e\n", data->likelihood[d]) ;
	}
	
	fclose(fp_rmax) ;
	fclose(fp_info) ;
	fclose(fp_likelihood) ;
}

void free_memory(struct max_data *data) {
	if (!data->refinement) {
		free(data->max_exp_p) ;
		free(data->p_sum) ;
	}
	free(data->info) ;
	free(data->likelihood) ;
	free(data->rmax) ;
	if (param->modes > 1)
		free(data->quat_norm) ;
	
	if (!data->within_openmp) {
		if (!data->refinement) {
			free(data->probab) ;
			free(data->max_exp) ;
		}
		free(data->u) ;
	}
	else {
		int d ;
		for (d = 0 ; d < det[0].num_det ; ++d)
			free(data->all_views[d]) ;
		free(data->all_views) ;
		if (param->need_scaling)
			free(data->scale) ;
		free(data->model) ;
		free(data->weight) ;
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

