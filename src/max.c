#include "emc.h"

double maximize() {
	int d, r, t, x ;
	double total = 0., rescale, avg_likelihood = 0. ;
	struct timeval t1, t2 ;
	
	// Increasing beta by a factor of 'beta_jump' every 'beta_period' iterations
	if (iteration % beta_period == 1 && iteration > 1)
		beta *= beta_jump ;
	
	gettimeofday(&t1, NULL) ;
	
	mutual_info = 0. ;
	
	// Allocate memory
	double **probab = malloc(num_rot_p * sizeof(double*)) ;
	double *u = malloc(num_rot_p * sizeof(double)) ;
	int *rmax = malloc(tot_num_data * sizeof(int)) ;
	double *max_exp = malloc(tot_num_data * sizeof(double)) ;
	double *max_exp_p = malloc(tot_num_data * sizeof(double)) ;
	double *mean_prob = malloc(tot_num_data * sizeof(double)) ;
	double *p_sum = malloc(tot_num_data * sizeof(double)) ;
	double *bestprob = malloc(tot_num_data * sizeof(double)) ;
	double *info = calloc(tot_num_data, sizeof(double)) ;
	double *likelihood = calloc(tot_num_data, sizeof(double)) ;
	for (d = 0 ; d < tot_num_data ; ++d) {
		max_exp_p[d] = -DBL_MAX ;
		p_sum[d] = 0. ;
		if (need_scaling)
			likelihood[d] = count[d]*log(scale[d]) - sum_fact[d] ;
		else
			likelihood[d] = -sum_fact[d] ;
	}
	
	memset(model2, 0, size*size*size*sizeof(double)) ;
	memset(inter_weight, 0, size*size*size*sizeof(double)) ;
	
	if (rank == 0) {
		gettimeofday(&t2, NULL) ;
		fprintf(stderr, "\tAlloc\t%f\n", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;
	}
	
	// Calculate rescale factor by calculating mean model value over detector
	#pragma omp parallel default(shared) private(r,t)
	{
		double *view = malloc(num_pix * sizeof(double)) ;
		
		#pragma omp for schedule(static,1) reduction(+:total)
		for (r = 0 ; r < num_rot_p ; ++r) {
			u[r] = 0. ;
			probab[r] = malloc(tot_num_data * sizeof(double)) ;
			if (probab[r] == NULL)
				fprintf(stderr, "Unable to allocate probab[%d]\n", r) ;
			
			// Second argument being 0. tells slice_gen to generate un-rescaled tomograms
			slice_gen(&quat[(r*num_proc + rank)*5], 0., view, model1, det) ;
			
			for (t = 0 ; t < num_pix ; ++t)
			if (mask[t] < 1)
				u[r] += view[t] ;
			
			total += quat[(r*num_proc + rank)*5 + 4] * u[r] ;
		}
		
		free(view) ;
	}
	
	MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	rescale = (double) tot_mean_count / total ;
	
	if (rank == 0) {
		gettimeofday(&t1, NULL) ;
		fprintf(stderr, "\trescale\t%f (= %.6e)\n", (double)(t1.tv_sec - t2.tv_sec) + (t1.tv_usec - t2.tv_usec) / 1000000., rescale) ;
	}
	
	// Sum over all pixels of model tomogram (data-independent part of probability)
	#pragma omp parallel for schedule(static,1) default(shared) private(r)
	for (r = 0 ; r < num_rot_p ; ++r)
		u[r] = log(quat[(r*num_proc + rank)*5 + 4]) - u[r] ;

	// Main loop: Calculate and update tomograms
	#pragma omp parallel default(shared) private(r,d,t,x)
	{
		int pixel, d_counter, omp_rank = omp_get_thread_num() ;
		long ones_counter = 0, multi_counter = 0 ;
		double sum, temp, *priv_scale = NULL ;
		double *view = malloc(num_pix * sizeof(double)) ;
		double *old_view = malloc(num_pix * sizeof(double)) ;
		int *priv_rmax = calloc(tot_num_data, sizeof(int)) ;
		double *priv_max = malloc(tot_num_data * sizeof(double)) ;
		double *priv_sum = malloc(tot_num_data * sizeof(double)) ;
		double *priv_model = calloc(size*size*size, sizeof(double)) ;
		double *priv_weight = calloc(size*size*size, sizeof(double)) ;
		double *priv_info = calloc(tot_num_data, sizeof(double)) ;
		double *priv_likelihood = calloc(tot_num_data, sizeof(double)) ;
		for (d = 0 ; d < tot_num_data ; ++d) {
			priv_max[d] = max_exp_p[d] ;
			priv_sum[d] = p_sum[d] ;
		}
		struct dataset *curr ;
		
		if (need_scaling)
			priv_scale = calloc(tot_num_data, sizeof(double)) ;
		
		double *prob = NULL ;
		
		// Calculate log-likelihood of frame, d having orientation, r (probab[r][d])
		// For each orientation in the MPI rank
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < num_rot_p ; ++r) {
			// Calculate rescaled log-tomogram (rescale !=0)
			slice_gen(&quat[(r*num_proc + rank)*5], 1., view, model1, det) ;
			
			prob = probab[r] ;
			d_counter = 0 ;
			curr = frames ;
			
			// Linked list of data sets from different files
			while (curr != NULL) {
				ones_counter = 0 ;
				multi_counter = 0 ;
				
				// For each frame in data set
				for (d = 0 ; d < curr->num_data ; ++d) {
					// check if frame is blacklisted
					if (blacklist[d_counter+d]) {
						ones_counter += curr->ones[d] ;
						multi_counter += curr->multi[d] ;
						continue ;
					}
					
					// need_scaling is for if we want to assume variable incident intensity
					if (need_scaling && (iteration > 1 || known_scale))
						prob[d_counter+d] = u[r] * scale[d_counter+d] ;
					else
						prob[d_counter+d] = u[r] * rescale ;
					
					// For each pixel with one photon
					for (t = 0 ; t < curr->ones[d] ; ++t) {
						pixel = curr->place_ones[ones_counter + t] ;
						if (mask[pixel] < 1)
							prob[d_counter+d] += view[pixel] ;
					}
					
					// For each pixel with count_multi photons
					for (t = 0 ; t < curr->multi[d] ; ++t) {
						pixel = curr->place_multi[multi_counter + t] ;
						if (mask[pixel] < 1)
							prob[d_counter+d] += curr->count_multi[multi_counter + t] * view[pixel] ;
					}
					
					// Note maximum log-likelihood for each frame among 'r's tested by this MPI rank and OMP rank
					if (prob[d_counter+d] > priv_max[d_counter+d]) {
						priv_max[d_counter+d] = prob[d_counter+d] ;
						priv_rmax[d_counter+d] = r*num_proc + rank ;
					}
					
					ones_counter += curr->ones[d] ;
					multi_counter += curr->multi[d] ;
				}
				
				d_counter += curr->num_data ;
				curr = curr->next ;
			}
			
			if ((r*num_proc + rank)%5000 == 0)
				fprintf(stderr, "\t\tFinished r = %d\n", r*num_proc + rank) ;
		}
		
		if (rank == 0 && omp_rank == 0) {
			gettimeofday(&t1, NULL) ;
			fprintf(stderr, "\tprob\t%f\n", (double)(t1.tv_sec - t2.tv_sec) + (t1.tv_usec - t2.tv_usec) / 1000000.) ;
		}
		
		// Calculate maximum log-likelihood for all frames among 'r's tested by this MPI rank
		#pragma omp critical(maxexp)
		{
			for (d = 0 ; d < tot_num_data ; ++d)
			if (priv_max[d] > max_exp_p[d]) {
				max_exp_p[d] = priv_max[d] ;
				rmax[d] = priv_rmax[d] ;
			}
		}
		#pragma omp barrier
		
		// Combine information about maximum log-likelihood among all 'r's
		if (omp_rank == 0) {
			MPI_Allreduce(max_exp_p, max_exp, tot_num_data, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) ;
			
			// Determine 'r' for which log-likelihood is maximum
			for (d = 0 ; d < tot_num_data ; ++d)
			if (max_exp[d] != max_exp_p[d] || max_exp_p[d] == -DBL_MAX)
				rmax[d] = -1 ;
			
			MPI_Allreduce(MPI_IN_PLACE, rmax, tot_num_data, MPI_INT, MPI_MAX, MPI_COMM_WORLD) ;
			
			if (rank == 0) {
				char fnamermax[999] ;
				sprintf(fnamermax, "%s/orientations/orientations_%.3d.bin", output_folder, iteration) ;
				FILE *fprmax = fopen(fnamermax, "w") ;
				fwrite(rmax, sizeof(int), tot_num_data, fprmax) ;
				fclose(fprmax) ;
			}
		}
		#pragma omp barrier
		
		// Calculate local normalization factor by summing over all orientations
		// max_exp is there to ensure that at least one orientation does not underflow
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < num_rot_p ; ++r) {
			prob = probab[r] ;
			
			for (d = 0 ; d < tot_num_data ; ++d)
				priv_sum[d] += exp(beta * (prob[d] - max_exp[d])) ;
		}
		
		// Combine information among all OMP ranks
		#pragma omp critical(psum)
		{
			for (d = 0 ; d < tot_num_data ; ++d)
				p_sum[d] += priv_sum[d] ;
		}
		#pragma omp barrier
		
		// Combine information among all MPI ranks
		if (omp_rank == 0)
			MPI_Allreduce(MPI_IN_PLACE, p_sum, tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
		#pragma omp barrier
		
		if (rank == 0 && omp_rank == 0) {
			gettimeofday(&t2, NULL) ;
			fprintf(stderr, "\tpsum\t%f\n", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;
		}
		
		// Calculate updated tomograms based on these probabilities
		#pragma omp for schedule(static,1)
		for (r = 0 ; r < num_rot_p ; ++r) {
			sum = 0. ;
			memset(view, 0, num_pix*sizeof(double)) ;
			
			prob = probab[r] ;
			d_counter = 0 ;
			if (iteration == num_iter + start_iter - 1 && merge_frames != NULL) {
				if (!rank && !r)
					fprintf(stderr, "Merging with different data file\n") ;
				curr = merge_frames ;
			}
			else
				curr = frames ;
			
			while (curr != NULL) {
				ones_counter = 0 ;
				multi_counter = 0 ;
				
				for (d = 0 ; d < curr->num_data ; ++d) {
					// check if frame is blacklisted
					if (blacklist[d_counter+d]) {
						ones_counter += curr->ones[d] ;
						multi_counter += curr->multi[d] ;
						continue ;
					}
					
					// Exponentiate log-likelihood and normalize to get probabilities
					temp = prob[d_counter+d] ;
					prob[d_counter+d] = exp(beta*(prob[d_counter+d] - max_exp[d_counter+d])) / p_sum[d_counter+d] ; 
//					priv_likelihood[d_counter+d] += prob[d_counter+d] * (temp - sum_fact[d_counter+d] + count[d_counter+d]*log(scale[d_counter+d])) ;
					priv_likelihood[d_counter+d] += prob[d_counter+d] * temp ;
					
					// Calculate denominator for update rule
					if (need_scaling) {
						sum += prob[d_counter+d] * scale[d_counter+d] ;
						// Calculate denominator for scale factor update rule
						if (iteration > 1)
							priv_scale[d_counter+d] -= prob[d_counter+d] * u[r] ;
						else
							priv_scale[d_counter+d] -= prob[d_counter+d] * u[r] * rescale ;
					}
					else
						sum += prob[d_counter+d] ; 
					
					// Skip if probability is very low (saves time)
					if (!(prob[d_counter+d] > PROB_MIN)) {
						ones_counter += curr->ones[d] ;
						multi_counter += curr->multi[d] ;
						continue ;
					}
					
					priv_info[d_counter+d] += prob[d_counter+d] * log(prob[d_counter+d] / quat[(r*num_proc + rank)*5 + 4]) ;
					
					// For all pixels with one photon
					for (t = 0 ; t < curr->ones[d] ; ++t) {
						pixel = curr->place_ones[ones_counter + t] ;
						if (mask[pixel] < 2)
							view[pixel] += prob[d_counter+d] ;
					}
					
					// For all pixels with count_multi photons
					for (t = 0 ; t < curr->multi[d] ; ++t) {
						pixel = curr->place_multi[multi_counter + t] ;
						if (mask[pixel] < 2)
							view[pixel] += curr->count_multi[multi_counter + t] * prob[d_counter+d] ;
					}
					
					ones_counter += curr->ones[d] ;
					multi_counter += curr->multi[d] ;
				}
				
				d_counter += curr->num_data ;
				curr = curr->next ;
			}
			
			if (alpha > 0.)
				slice_gen(&quat[(r*num_proc + rank)*5], rescale, old_view, model1, det) ;
			
			// If no data frame has any probability for this orientation, don't merge
			// Otherwise divide the updated tomogram by the sum over all probabilities and merge
			if (sum > 0.) {
				for (t = 0 ; t < num_pix ; ++t) {
					view[t] /= sum ;
					
					if (alpha > 0.)
						old_view[t] = (1.-alpha) * view[t] + alpha * old_view[t] ;
				}
				
				/*
				for (d = 0 ; d < tot_num_data ; ++d)
				if (r*num_proc + rank == rmax[d])
				for (t = 0 ; t < num_pix ; ++t)
						priv_likelihood[d] -= prob[d] * log(gsl_sf_bessel_I0_scaled(2.*view[t]*scale[d])) ;
				*/
				
				if (alpha == 0.)
					slice_merge(&quat[(r*num_proc + rank)*5], view, priv_model, priv_weight, det) ;
			}
			
			if (alpha > 0.)
				slice_merge(&quat[(r*num_proc + rank)*5], old_view, priv_model, priv_weight, det) ;
			
			free(probab[r]) ;
		}
		
		// Combine merges from different OMP ranks (MPI ranks are combined in recon.c)
		#pragma omp critical(model)
		{
			for (x = 0 ; x < size * size * size ; ++x) {
				model2[x] += priv_model[x] ;
				inter_weight[x] += priv_weight[x] ;
			}
		}
		
		// Combine mutual info from different OMP ranks
		#pragma omp critical(like_info)
		{
			for (d = 0 ; d < tot_num_data ; ++d) {
				likelihood[d] += priv_likelihood[d] ;
				info[d] += priv_info[d] ;
			}
		}
		#pragma omp barrier
		
		if (need_scaling) {
			if (omp_rank == 0)
				memset(scale, 0, tot_num_data * sizeof(double)) ;
			
			// Combine information for scale factors from each OMP rank
			#pragma omp barrier
			#pragma omp critical(scale)
			{
				for (d = 0 ; d < tot_num_data ; ++d)
				if (!blacklist[d])
					scale[d] += priv_scale[d] ;
			}
			#pragma omp barrier
			
			free(priv_scale) ;
			
			// Combine information for scale factors from all MPI ranks
			if (omp_rank == 0)
				MPI_Allreduce(MPI_IN_PLACE, scale, tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
		}
		
		free(priv_model) ;
		free(priv_weight) ;
		free(priv_sum) ;
		free(priv_max) ;
		free(priv_info) ;
		free(priv_likelihood) ;
		free(view) ;
		free(old_view) ;
	}

	if (rank == 0) {
		gettimeofday(&t1, NULL) ;
		fprintf(stderr, "\tUpdate\t%f\n", (double)(t1.tv_sec - t2.tv_sec) + (t1.tv_usec - t2.tv_usec) / 1000000.) ;
	}
	
	free(probab) ;
	free(p_sum) ;
	free(rmax) ;
	free(mean_prob) ;
	free(max_exp) ;
	free(max_exp_p) ;
	free(u) ;
	free(bestprob) ;
	
	// Combine mutual info and likelihood from all MPI ranks
	MPI_Allreduce(MPI_IN_PLACE, likelihood, tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	MPI_Allreduce(MPI_IN_PLACE, info, tot_num_data, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
	
	// Calculate updated scale factor using count[d] (total photons in frame d)
	if (need_scaling) {
		for (d = 0 ; d < tot_num_data ; ++d)
		if (!blacklist[d])
			scale[d] = count[d] / scale[d] ;
		
		if (rank == 0) {
			char fname[100] ;
			sprintf(fname, "%s/scale/scale_%.3d.dat", output_folder, iteration) ;
			FILE *fp_scale = fopen(fname, "w") ;
			for (d = 0 ; d < tot_num_data ; ++d)
				fprintf(fp_scale, "%.6e\n", scale[d]) ;
			fclose(fp_scale) ;
		}
	}
	
	// Print frame-by-frame mutual information and likelihood to file
	for (d = 0 ; d < tot_num_data ; ++d)
	if (!blacklist[d]) {
		mutual_info += info[d] ;
		avg_likelihood += likelihood[d] ;
	}
	
	if (rank == 0) {
		char fname[999] ;
		sprintf(fname, "%s/mutualInfo/info_%.3d.dat", output_folder, iteration) ;
		FILE *fp_info = fopen(fname, "w") ;
		sprintf(fname, "%s/likelihood/likelihood_%.3d.dat", output_folder, iteration) ;
		FILE *fp_likelihood = fopen(fname, "w") ;
		
		for (d = 0 ; d < tot_num_data ; ++d) {
			fprintf(fp_info, "%.6e\n", info[d]) ;
			fprintf(fp_likelihood, "%.6e\n", likelihood[d]) ;
		}
		
		fclose(fp_info) ;
		fclose(fp_likelihood) ;
	}
	
	mutual_info /= (tot_num_data - num_blacklist) ;
	avg_likelihood /= (tot_num_data - num_blacklist) ;
	free(info) ;
	free(likelihood) ;
	
	return avg_likelihood ;
}
