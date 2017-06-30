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

static double *u, *max_exp_p, *max_exp, *p_sum, *info, *likelihood ;
static int *rmax ;
static struct timeval t1, t2 ;

static void allocate_memory(double***) ;
static double calculate_rescale() ;
static void calculate_prob(int, double*, int*, double*) ;
static void normalize_prob(double**, double*, int*) ;
static double update_tomogram(int, double*, double*, double*) ;
static void merge_tomogram(int, double, double*, double*, double*) ;
static void combine_information(double*, double*, double*) ;
static void print_time(char*, char*, int) ;
static void free_memory(double**) ;

double maximize() {
	int d, r ;
	long vol = iter->size * iter->size * iter->size ;
	double **probab, avg_likelihood = 0. ;
	gettimeofday(&t1, NULL) ;
	iter->mutual_info = 0. ;

	allocate_memory(&probab) ;

	// Sum over all pixels of model tomogram (data-independent part of probability)
	iter->rescale = calculate_rescale() ;
	fprintf(stderr, "\trescale = %.6e\n", iter->rescale) ;

	// Main loop: Calculate probabilities and update tomograms
	#pragma omp parallel default(shared) private(r,d)
	{
		int omp_rank = omp_get_thread_num() ;
		// priv_data = {priv_likelihood, priv_info, priv_scale}
		double sum, *priv_data = NULL ;
		double *view = malloc(det->num_pix * sizeof(double)) ;
		int *priv_rmax = calloc(frames->tot_num_data, sizeof(int)) ;
		double *priv_max = malloc(frames->tot_num_data * sizeof(double)) ;
		double *priv_model = calloc(vol, sizeof(double)) ;
		double *priv_weight = calloc(vol, sizeof(double)) ;
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			priv_max[d] = max_exp_p[d] ;
		
		if (param.need_scaling)
			priv_data = calloc(3 * frames->tot_num_data, sizeof(double)) ;
		else
			priv_data = calloc(2 * frames->tot_num_data, sizeof(double)) ;

		#pragma omp for schedule(static,1)
		for (r = 0 ; r < quat->num_rot_p ; ++r) {
			probab[r] = malloc(frames->tot_num_data * sizeof(double)) ;
			calculate_prob(r, priv_max, priv_rmax, probab[r]) ;
		}
		print_time("prob", "", rank == 0 && omp_rank == 0) ;
		
		normalize_prob(probab, priv_max, priv_rmax) ;
		print_time("psum", "", rank == 0 && omp_rank == 0) ;

		#pragma omp for schedule(static,1)
		for (r = 0 ; r < quat->num_rot_p ; ++r) {
			sum = update_tomogram(r, probab[r], priv_data, view) ;
			merge_tomogram(r, sum, view, priv_model, priv_weight) ;
			
			free(probab[r]) ;
		}

		// Combine information from different OpenMP ranks
		// This function (and the associated private arrays) will be unnecessary with
		// OpenMP 4.5 support available in GCC 6.1+ or ICC 17.0+
		combine_information(priv_data, priv_model, priv_weight) ;
		
		free(view) ;
	}
	print_time("Update", "", rank == 0) ;

	// Combine 3D volumes from all MPI ranks
	if (rank) {
		MPI_Reduce(iter->model2, iter->model2, vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
		MPI_Reduce(iter->inter_weight, iter->inter_weight, vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
	}
	else {
		MPI_Reduce(MPI_IN_PLACE, iter->model2, vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
		MPI_Reduce(MPI_IN_PLACE, iter->inter_weight, vol, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
	}
	
	// Combine mutual info and likelihood from all MPI ranks
	MPI_Allreduce(MPI_IN_PLACE, likelihood, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	MPI_Allreduce(MPI_IN_PLACE, info, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;

	// Calculate updated scale factor using count[d] (total photons in frame d)
	if (param.need_scaling) {
		// Combine scale factor information from all MPI ranks
		MPI_Allreduce(MPI_IN_PLACE, iter->scale, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
		
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (!frames->blacklist[d])
			iter->scale[d] = frames->count[d] / iter->scale[d] ;
		
		if (rank == 0) {
			char fname[100] ;
			sprintf(fname, "%s/scale/scale_%.3d.dat", param.output_folder, param.iteration) ;
			FILE *fp_scale = fopen(fname, "w") ;
			for (d = 0 ; d < frames->tot_num_data ; ++d)
				fprintf(fp_scale, "%.6e\n", iter->scale[d]) ;
			fclose(fp_scale) ;
		}
	}

	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (!frames->blacklist[d]) {
		iter->mutual_info += info[d] ;
		avg_likelihood += likelihood[d] ;
	}
	
	// Print frame-by-frame mutual information, likelihood, and most likely orientations to file
	if (rank == 0) {
		char fname[1024] ;
		sprintf(fname, "%s/mutualInfo/info_%.3d.dat", param.output_folder, param.iteration) ;
		FILE *fp_info = fopen(fname, "w") ;
		sprintf(fname, "%s/likelihood/likelihood_%.3d.dat", param.output_folder, param.iteration) ;
		FILE *fp_likelihood = fopen(fname, "w") ;
		sprintf(fname, "%s/orientations/orientations_%.3d.bin", param.output_folder, param.iteration) ;
		FILE *fp_rmax = fopen(fname, "w") ;
		FILE *fp_sum = fopen("data/psum.bin", "wb") ;
		
		fwrite(p_sum, sizeof(double), frames->tot_num_data, fp_sum) ;
		fwrite(rmax, sizeof(int), frames->tot_num_data, fp_rmax) ;
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			fprintf(fp_info, "%.6e\n", info[d]) ;
			fprintf(fp_likelihood, "%.6e\n", likelihood[d]) ;
		}
		
		fclose(fp_rmax) ;
		fclose(fp_info) ;
		fclose(fp_likelihood) ;
		fclose(fp_sum) ;
	}
	
	iter->mutual_info /= (frames->tot_num_data - frames->num_blacklist) ;
	avg_likelihood /= (frames->tot_num_data - frames->num_blacklist) ;
	
	free_memory(probab) ;
	
	return avg_likelihood ;
}

void allocate_memory(double ***probab) {
	int d ;
	long vol = iter->size * iter->size * iter->size ;
	
	// Allocate memory
	*probab = malloc(quat->num_rot_p * sizeof(double*)) ;
	u = malloc(quat->num_rot_p * sizeof(double)) ;
	rmax = malloc(frames->tot_num_data * sizeof(int)) ;
	max_exp = malloc(frames->tot_num_data * sizeof(double)) ;
	max_exp_p = malloc(frames->tot_num_data * sizeof(double)) ;
	p_sum = calloc(frames->tot_num_data, sizeof(double)) ;
	info = calloc(frames->tot_num_data, sizeof(double)) ;
	likelihood = calloc(frames->tot_num_data, sizeof(double)) ;
	for (d = 0 ; d < frames->tot_num_data ; ++d) {
		max_exp_p[d] = -DBL_MAX ;
		if (param.need_scaling)
			likelihood[d] = frames->count[d]*log(iter->scale[d]) - frames->sum_fact[d] ;
		else
			likelihood[d] = -frames->sum_fact[d] ;
	}
	
	memset(iter->model2, 0, vol*sizeof(double)) ;
	memset(iter->inter_weight, 0, vol*sizeof(double)) ;
	print_time("Alloc", "", rank == 0) ;
}

double calculate_rescale() {
	int r, t ;
	double total = 0. ;
	
	// Calculate rescale factor by calculating mean model value over detector
	#pragma omp parallel default(shared) private(r, t)
	{
		double *view = malloc(det->num_pix * sizeof(double)) ;
		
		#pragma omp for schedule(static,1) reduction(+:total)
		for (r = 0 ; r < quat->num_rot_p ; ++r) {
			u[r] = 0. ;
			
			// Second argument being 0. tells slice_gen to generate un-rescaled tomograms
			slice_gen(&quat->quat[(r*num_proc + rank)*5], 0., view, iter->model1, iter->size, det) ;
			
			for (t = 0 ; t < det->num_pix ; ++t)
			if (det->mask[t] < 1)
				u[r] += view[t] ;
			
			total += quat->quat[(r*num_proc + rank)*5 + 4] * u[r] ;
		}
		
		free(view) ;
	}
	
	MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	#pragma omp parallel for schedule(static,1) default(shared) private(r)
	for (r = 0 ; r < quat->num_rot_p ; ++r) {
		u[r] = log(quat->quat[(r*num_proc + rank)*5 + 4]) - u[r] ;
	}
	
	char res_string[1024] ;
	sprintf(res_string, "(= %.6e)", frames->tot_mean_count / total) ;
	print_time("rescale", res_string, rank == 0) ;
	
	return frames->tot_mean_count / total ;
}

void calculate_prob(int r, double *max, int *rmax, double *prob) {
	int t, d, d_counter = 0, pixel ;
	long ones_counter, multi_counter ;
	struct dataset *curr = frames ;
	double *view = malloc(det->num_pix * sizeof(double)) ;
	
	slice_gen(&quat->quat[(r*num_proc + rank)*5], 1., view, iter->model1, iter->size, det) ;
	
	// Linked list of data sets from different files
	while (curr != NULL) {
		ones_counter = 0 ;
		multi_counter = 0 ;
		
		// For each frame in data set
		for (d = 0 ; d < curr->num_data ; ++d) {
			// check if frame is blacklisted
			if (frames->blacklist[d_counter+d]) {
				if (curr->type == 0) {
					ones_counter += curr->ones[d] ;
					multi_counter += curr->multi[d] ;
				}
				continue ;
			}
			
			if (curr->type < 2) {
				// need_scaling is for if we want to assume variable incident intensity
				if (param.need_scaling && (param.iteration > 1 || param.known_scale))
					prob[d_counter+d] = u[r] * iter->scale[d_counter+d] ;
				else
					prob[d_counter+d] = u[r] * iter->rescale ;
			}
			else {
				prob[d_counter+d] = 0. ;
			}
			
			if (curr->type == 0) {
				// For each pixel with one photon
				for (t = 0 ; t < curr->ones[d] ; ++t) {
					pixel = curr->place_ones[ones_counter + t] ;
					if (det->mask[pixel] < 1)
						prob[d_counter+d] += view[pixel] ;
				}
				
				// For each pixel with count_multi photons
				for (t = 0 ; t < curr->multi[d] ; ++t) {
					pixel = curr->place_multi[multi_counter + t] ;
					if (det->mask[pixel] < 1)
						prob[d_counter+d] += curr->count_multi[multi_counter + t] * view[pixel] ;
				}
			}
			else if (curr->type == 1) {
				for (t = 0 ; t < det->num_pix ; ++t)
				if (det->mask[t] < 1)
					prob[d_counter+d] += curr->int_frames[d*curr->num_pix + t] * view[t] ;
			}
			else if (curr->type == 2) { // Gaussian EMC for double precision data without scaling
				for (t = 0 ; t < det->num_pix ; ++t)
				if (det->mask[t] < 1)
					prob[d_counter+d] -= pow(curr->frames[d*curr->num_pix + t] - view[t]*iter->rescale, 2.) ;
			}
			
			// Note maximum log-likelihood for each frame among 'r's tested by this MPI rank and OMP rank
			if (prob[d_counter+d] > max[d_counter+d]) {
				max[d_counter+d] = prob[d_counter+d] ;
				rmax[d_counter+d] = r*num_proc + rank ;
			}
			
			if (curr->type == 0) {
				ones_counter += curr->ones[d] ;
				multi_counter += curr->multi[d] ;
			}
		}
		
		d_counter += curr->num_data ;
		curr = curr->next ;
	}
	
	free(view) ;
	
	if ((r*num_proc + rank)%5000 == 0)
		fprintf(stderr, "\t\tFinished r = %d\n", r*num_proc + rank) ;
}

void normalize_prob(double **prob, double *priv_max, int *priv_rmax) {
	int r, d, omp_rank = omp_get_thread_num() ;
	double *priv_sum = malloc(frames->tot_num_data * sizeof(double)) ;
	
	#pragma omp critical(maxexp)
	{
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (priv_max[d] > max_exp_p[d]) {
			max_exp_p[d] = priv_max[d] ;
			rmax[d] = priv_rmax[d] ;
		}
	}
	#pragma omp barrier
	
	if (omp_rank == 0) {
		MPI_Allreduce(max_exp_p, max_exp, frames->tot_num_data, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) ;
		
		// Determine 'r' for which log-likelihood is maximum
		for (d = 0 ; d < frames->tot_num_data ; ++d)
		if (max_exp[d] != max_exp_p[d] || max_exp_p[d] == -DBL_MAX)
			rmax[d] = -1 ;
		
		MPI_Allreduce(MPI_IN_PLACE, rmax, frames->tot_num_data, MPI_INT, MPI_MAX, MPI_COMM_WORLD) ;
	}
	#pragma omp barrier
	
	#pragma omp for schedule(static,1)
	for (r = 0 ; r < quat->num_rot_p ; ++r)
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (frames->type < 2)
		priv_sum[d] += exp(param.beta * (prob[r][d] - max_exp[d])) ;
	else
		priv_sum[d] += exp(param.beta * (prob[r][d] - max_exp[d]) / 2. / param.sigmasq) ;
	
	#pragma omp critical(psum)
	{
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			p_sum[d] += priv_sum[d] ;
	}
	#pragma omp barrier
	
	if (omp_rank == 0)
		MPI_Allreduce(MPI_IN_PLACE, p_sum, frames->tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	#pragma omp barrier
	
	free(priv_max) ;
	free(priv_rmax) ;
	free(priv_sum) ;
}

double update_tomogram(int r, double *prob, double *priv_data, double *view) {
	int t, d, d_counter = 0, pixel ;
	double temp, sum = 0. ;
	long ones_counter, multi_counter ;
	struct dataset *curr ;
	
	if (merge_frames != NULL) {
		if (!rank && !r)
			fprintf(stderr, "Merging with different data file: %s\n", merge_frames->filename) ;
		curr = merge_frames ;
	}
	else
		curr = frames ;
	
	memset(view, 0, det->num_pix*sizeof(double)) ;
	
	while (curr != NULL) {
		ones_counter = 0 ;
		multi_counter = 0 ;
		
		for (d = 0 ; d < curr->num_data ; ++d) {
			// check if frame is blacklisted
			if (frames->blacklist[d_counter+d]) {
				if (curr->type == 0) {
					ones_counter += curr->ones[d] ;
					multi_counter += curr->multi[d] ;
				}
				continue ;
			}
			
			// Exponentiate log-likelihood and normalize to get probabilities
			temp = prob[d_counter+d] ;
			if (frames->type < 2)
				prob[d_counter+d] = exp(param.beta*(prob[d_counter+d] - max_exp[d_counter+d])) / p_sum[d_counter+d] ; 
			else
				prob[d_counter+d] = exp(param.beta * (prob[d_counter+d] - max_exp[d_counter+d]) / 2. / param.sigmasq) / p_sum[d_counter+d] ;
//			priv_data[d_counter+d] += prob[d_counter+d] * (temp - frames->sum_fact[d_counter+d] + frames->count[d_counter+d]*log(iter->scale[d_counter+d])) ;
			priv_data[d_counter+d] += prob[d_counter+d] * temp ;
			
			// Calculate denominator for update rule
			if (param.need_scaling) {
				sum += prob[d_counter+d] * iter->scale[d_counter+d] ;
				// Calculate denominator for scale factor update rule
				if (param.iteration > 1)
					priv_data[2*frames->tot_num_data + d_counter+d] -= prob[d_counter+d] * u[r] ;
				else
					priv_data[2*frames->tot_num_data + d_counter+d] -= prob[d_counter+d] * u[r] * iter->rescale ;
			}
			else
				sum += prob[d_counter+d] ; 
			
			// Skip if probability is very low (saves time)
			if (!(prob[d_counter+d] > PROB_MIN)) {
				if (curr->type == 0) {
					ones_counter += curr->ones[d] ;
					multi_counter += curr->multi[d] ;
				}
				continue ;
			}
			
			// Calculate mutual information of probability distribution
			priv_data[frames->tot_num_data + d_counter+d] += prob[d_counter+d] * log(prob[d_counter+d] / quat->quat[(r*num_proc + rank)*5 + 4]) ;
			//priv_data[frames->tot_num_data + d_counter+d] -= prob[d_counter+d] * log(prob[d_counter+d]) ;
			
			if (curr->type == 0) {
				// For all pixels with one photon
				for (t = 0 ; t < curr->ones[d] ; ++t) {
					pixel = curr->place_ones[ones_counter + t] ;
					if (det->mask[pixel] < 2)
						view[pixel] += prob[d_counter+d] ;
				}
				
				// For all pixels with count_multi photons
				for (t = 0 ; t < curr->multi[d] ; ++t) {
					pixel = curr->place_multi[multi_counter + t] ;
					if (det->mask[pixel] < 2)
						view[pixel] += curr->count_multi[multi_counter + t] * prob[d_counter+d] ;
				}
			}
			else if (curr->type == 1) {
				for (t = 0 ; t < curr->num_pix ; ++t)
					view[t] += curr->int_frames[d*curr->num_pix + t] * prob[d_counter+d] ;
			}
			else if (curr->type == 2) { // Gaussian EMC update without scaling
				for (t = 0 ; t < curr->num_pix ; ++t)
				if (det->mask[t] < 2)
					view[t] += curr->frames[d*curr->num_pix + t] * prob[d_counter+d] ;
			}
			
			if (curr->type == 0) {
				ones_counter += curr->ones[d] ;
				multi_counter += curr->multi[d] ;
			}
		}
		
		d_counter += curr->num_data ;
		curr = curr->next ;
	}
	
	return sum ;
}

void merge_tomogram(int r, double sum, double *view, double *model, double *weight) {
	int t ;
	
	// If no data frame has any probability for this orientation, don't merge
	// Otherwise divide the updated tomogram by the sum over all probabilities and merge
	if (sum > 0.) {
		for (t = 0 ; t < det->num_pix ; ++t)
			view[t] /= sum ;
		
		slice_merge(&quat->quat[(r*num_proc + rank)*5], view, model, weight, iter->size, det) ;
	}
}

void combine_information(double *priv_data, double *priv_model, double *priv_weight) {
	int d, omp_rank = omp_get_thread_num() ;
	long x, vol = iter->size * iter->size * iter->size ;
	
	#pragma omp critical(model)
	{
		for (x = 0 ; x < vol ; ++x) {
			iter->model2[x] += priv_model[x] ;
			iter->inter_weight[x] += priv_weight[x] ;
		}
	}
	
	#pragma omp critical(like_info)
	{
		for (d = 0 ; d < frames->tot_num_data ; ++d) {
			likelihood[d] += priv_data[d] ;
			info[d] += priv_data[frames->tot_num_data + d] ;
		}
	}
	
	if (param.need_scaling) {
		if (omp_rank == 0)
			memset(iter->scale, 0, frames->tot_num_data * sizeof(double)) ;
		#pragma omp barrier
		
		#pragma omp critical(scale)
		{
			for (d = 0 ; d < frames->tot_num_data ; ++d)
			if (!frames->blacklist[d])
				iter->scale[d] += priv_data[2*frames->tot_num_data + d] ;
		}
	}
	
	free(priv_model) ;
	free(priv_weight) ;
	free(priv_data) ;
}

void free_memory(double **probab) {
	free(probab) ;
	
	free(u) ;
	free(max_exp_p) ;
	free(max_exp) ;
	free(p_sum) ;
	free(info) ;
	free(likelihood) ;
	free(rmax) ;
}

void print_time(char *pre_tag, char *post_tag, int flag) {
	if (!flag)
		return ;
	
	double diff ;
	double time_1 = t1.tv_sec + t1.tv_usec*1.e-6 ;
	double time_2 = t2.tv_sec + t2.tv_usec*1.e-6 ;
	
	if (time_1 > time_2) {
		gettimeofday(&t2, NULL) ;
		diff = t2.tv_sec + t2.tv_usec*1.e-6 - time_1 ;
	}
	else {
		gettimeofday(&t1, NULL) ;
		diff = t1.tv_sec + t1.tv_usec*1.e-6 - time_2 ;
	}
	
	fprintf(stderr, "\t%s\t%f %s\n", pre_tag, diff, post_tag) ;
}

