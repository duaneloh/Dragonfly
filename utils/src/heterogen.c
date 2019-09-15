#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <hdf5.h>

#include "../../src/utils.h"
#include "../../src/params.h"
#include "../../src/dataset.h"
#include "../../src/detector.h"
#include "../../src/iterate.h"
#include "../../src/interp.h"
#include "../../src/quat.h"

struct params *param ;
struct detector *det ;
struct dataset *frames ;
struct iterate *iter ;
struct rotation *quat ;
int *rots, *sort_args ;

static void print_recon_time(char *message, struct timeval *time_1, struct timeval *time_2, int rank) {
	if (!rank) {
		gettimeofday(time_2, NULL) ;
		fprintf(stderr, "%s: %f s\n", message, (double)(time_2->tv_sec - time_1->tv_sec) + 1.e-6*(time_2->tv_usec - time_1->tv_usec)) ;
	}
}

int parse_arguments(int argc, char *argv[], int *num_threads, char *config_fname) {
	int c, num_iter = -1 ;
	extern char *optarg ;
	extern int optind ;
	strcpy(config_fname, "config.ini") ;
	
	while (optind < argc) {
		if ((c = getopt(argc, argv, "c:t:")) != -1) {
			switch (c) {
				case 't':
					*num_threads = atoi(optarg) ;
					omp_set_num_threads(*num_threads) ;
					break ;
				case 'c':
					strcpy(config_fname, optarg) ;
					break ;
			}
		}
		else {
			num_iter = atoi(argv[optind]) ;
			optind++ ;
		}
	}
	
	if (num_iter == -1) {
		fprintf(stderr, "Format: %s [-c config_fname] [-t num_threads] num_iter\n", argv[0]) ;
		fprintf(stderr, "Default: -c config.ini -t %d\n", omp_get_max_threads()) ;
		fprintf(stderr, "Missing <num_iter>\n") ;
		return -1 ;
	}
	fprintf(stderr, "Doing %d iteration(s) using %s\n", num_iter, config_fname) ;
	
	return num_iter ;
}

int argsort_rots(const void *val1, const void *val2) {
	int d1 = *((int*) val1) ;
	int d2 = *((int*) val2) ;
	return rots[d1] - rots[d2] ;
}

int parse_rots(char *fname) {
	FILE *fp = fopen(fname, "rb") ;
	if (fp == NULL) {
		fprintf(stderr, "Unable to open rots_file %s\n", fname) ;
		return 1 ;
	}
	
	rots = malloc(frames->tot_num_data * sizeof(int)) ;
	
#ifndef WITH_HDF5
	fread(rots, sizeof(int), frames->tot_num_data, fp) ;
	fclose(fp) ;
#else // WITH_HDF5
	fclose(fp) ;
	
	hid_t file, dset, dspace ;
	hsize_t npoints ;
	
	file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT) ;
	dset = H5Dopen(file, "/orientations", H5P_DEFAULT) ;
	dspace = H5Dget_space(dset) ;
	npoints = H5Sget_simple_extent_npoints(dspace) ;
	if (npoints != iter->tot_num_data) {
		fprintf(stderr, "Number of frames in 'orientations' dataset does not match.\n") ;
		if (npoints < iter->tot_num_data) {
			fprintf(stderr, "Reading orientations into first %lld frames, others unity\n", npoints) ;
			dspace = H5S_ALL ;
		}
		else {
			fprintf(stderr, "Reading only first %d orientations out of %lld\n", iter->tot_num_data, npoints) ;
			npoints = iter->tot_num_data ;
			dspace = H5Screate_simple(1, &npoints, NULL) ;
		}
		//fprintf(stderr, "Defaulting to uniform scale factors\n") ;
		//return 0 ;
	}
	H5Dread(dset, H5T_STD_I32LE, dspace, dspace, H5P_DEFAULT, rots) ;
	H5Dclose(dset) ;
	H5Fclose(file) ;
#endif
	
	// Sort indices by orientation within each file
	int d ;
	struct dataset *curr = frames ;
	sort_args = malloc(iter->tot_num_data * sizeof(int)) ;
	for (d = 0 ; d < iter->tot_num_data ; ++d)
		sort_args[d] = d ;
	while (curr != NULL) {
		qsort(&(sort_args[curr->num_data_prev]), curr->tot_num_data, sizeof(int), argsort_rots) ;
		curr = curr->next ;
	}
	
	return 0 ;
}

void generate_models() {
	long m, x ;
	const gsl_rng_type *T ;
	gsl_rng_env_setup() ;
	T = gsl_rng_default ;
	gsl_rng *rng = gsl_rng_alloc(T) ;
	
	for (m = 1 ; m < iter->modes ; ++m)
	for (x = 0 ; x < iter->vol ; ++x)
		iter->model1[m*iter->vol + x] = iter->model1[x] * (0.4 * gsl_rng_uniform(rng) + 0.8) ; 
	for (x = 0 ; x < iter->vol ; ++x)
		iter->model1[x] *= (0.4 * gsl_rng_uniform(rng) + 0.8) ; 
	
	gsl_rng_free(rng) ;
}

void save_initial_iterate() {
#ifndef WITH_HDF5
	FILE *fp ;
	char fname[2048] ;
	long tot_vol = iter->modes * iter->vol ;
	
	sprintf(fname, "%s/output/intens_000.bin", param->output_folder) ;
	fp = fopen(fname, "w") ;
	fwrite(iter->model1, sizeof(double), tot_vol, fp) ;
	fclose(fp) ;
	
	if (param->need_scaling) {
		sprintf(fname, "%s/scale/scale_000.dat", param->output_folder) ;
		fp = fopen(fname, "w") ;
		for (d = 0 ; d < iter->tot_num_data ; ++d)
			fprintf(fp, "%.6e\n", iter->scale[d]) ;
		fclose(fp) ;
		fprintf(stderr, "Written initial scale factors to %s\n", fname) ;
	}
	
#else // WITH_HDF5
	
	hid_t file, dset, dspace ;
	char name[2048] ;
	hsize_t out_size3d[4], out_size2d[3], len[1] ;
	len[0] = frames->tot_num_data ;
	out_size3d[0] = iter->modes ;
	out_size3d[1] = iter->size ;
	out_size3d[2] = iter->size ;
	out_size3d[3] = iter->size ;
	out_size2d[0] = iter->modes ;
	out_size2d[1] = iter->size ;
	out_size2d[2] = iter->size ;
	
	sprintf(name, "%s/output_000.h5", param->output_folder) ;
	file = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) ;
	
	if (param->recon_type == RECON2D || param->recon_type == RECONRZ)
		dspace = H5Screate_simple(3, out_size2d, NULL) ;
	else
		dspace = H5Screate_simple(4, out_size3d, NULL) ;
	dset = H5Dcreate(file, "/intens", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
	H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, iter->model1) ;
	H5Dclose(dset) ;
	H5Sclose(dspace) ;
	
	if (param->need_scaling) {
		dspace = H5Screate_simple(1, len, NULL) ;
		dset = H5Dcreate(file, "scale", H5T_STD_I32LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
		H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, iter->scale) ;
		H5Dclose(dset) ;
		H5Sclose(dspace) ;
	}
	
	fprintf(stderr, "Written initial iterate to %s\n", name) ;
	H5Fclose(file) ;
#endif // WITH_HDF5
}

int setup(char *config_fname) {
	double qmax ;
	FILE *fp ;
	struct timeval t1, t2 ;
	char line[1024], section_name[1024], *token ;
	char input_fname[1024], scale_fname[1024], rots_fname[1024] ;

	input_fname[0] = '\0' ;
	scale_fname[0] = '\0' ;
	rots_fname[0] = '\0' ;
	
	gettimeofday(&t1, NULL) ;
	
	param = calloc(1, sizeof(struct params)) ;
	iter = calloc(1, sizeof(struct iterate)) ;
	quat = malloc(sizeof(struct rotation)) ;
	frames = malloc(sizeof(struct dataset)) ;
	param->rank = 0 ;
	param->num_proc = 1 ;
	iter->size = -1 ;
	
	fp = fopen(config_fname, "r") ;
	if (fp == NULL) {
		fprintf(stderr, "Config file %s not found.\n", config_fname) ;
		return 1 ;
	}
	while (fgets(line, 1024, fp) != NULL) {
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "heterogen") == 0) {
			if (strcmp(token, "start_model_file") == 0)
				strcpy(input_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "scale_file") == 0)
				strcpy(scale_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "rots_file") == 0)
				strcpy(rots_fname, strtok(NULL, " =\n")) ;
		}
	}
	fclose(fp) ;
	
	params_from_config(config_fname, "heterogen", param) ;
	if ((qmax = detector_from_config(config_fname, "heterogen", &det, 1)) < 0.)
		return 1 ;
	if (quat_from_config(config_fname, "heterogen", quat))
		return 1 ;
	if (data_from_config(config_fname, "heterogen", "in", det, frames))
		return 1 ;
	blacklist_from_config(config_fname, "heterogen", frames) ;
	
	iter->modes = param->modes + param->nonrot_modes ;
	iter->tot_num_data = frames->tot_num_data ;
	calculate_size(qmax, iter) ;
	parse_input(input_fname, 1., 0, RECON3D, iter) ;
	if (param->need_scaling) {
		calc_scale(frames, det, iter) ;
		param->known_scale = parse_scale(scale_fname, iter->scale, iter) ;
		if (param->known_scale == 0) {
			fprintf(stderr, "Need intial scale factors (scale_file) if need_scaling = 1\n") ;
			return 1 ;
		}
	}
	if (parse_rots(rots_fname))
		return 1 ;
	calc_beta(frames, param) ;
	
	//generate_models() ;
	save_initial_iterate() ;
	
	gettimeofday(&t2, NULL) ;
	fprintf(stderr, "Completed setup: %f s\n", (double)(t2.tv_sec - t1.tv_sec) + 1.e-6*(t2.tv_usec - t1.tv_usec)) ;
	
	return 0 ;
}

void write_log_file_header(int num_threads) {
	FILE *fp = fopen(param->log_fname, "w") ;
	fprintf(fp, "Heterogenize model\n\n") ;
	fprintf(fp, "Data parameters:\n") ;
	if (frames->num_blacklist == 0)
		fprintf(fp, "\tnum_data = %d\n\tmean_count = %f\n\n", frames->tot_num_data, frames->tot_mean_count) ;
	else
		fprintf(fp, "\tnum_data = %d/%d\n\tmean_count = %f\n\n", frames->tot_num_data-frames->num_blacklist, frames->tot_num_data, frames->tot_mean_count) ;
	fprintf(fp, "System size:\n") ;
	fprintf(fp, "\tnum_rot = %d\n\tnum_pix = %d/%d\n\t", quat->num_rot, det->rel_num_pix, det->num_pix) ;
	if (param->recon_type == RECON3D)
		fprintf(fp, "system_volume = %d X %ld X %ld X %ld\n\n", iter->modes, iter->size, iter->size, iter->size) ;
	else if (param->recon_type == RECON2D || param->recon_type == RECONRZ)
		fprintf(fp, "system_volume = %d X %ld X %ld\n\n", iter->modes, iter->size, iter->size) ;
	fprintf(fp, "Reconstruction parameters:\n") ;
	fprintf(fp, "\tnum_threads = %d\n\tnum_proc = %d\n\talpha = %.6f\n\tbeta = %.6f\n\tneed_scaling = %s", 
			num_threads, 
			param->num_proc, 
			param->alpha, 
			param->beta_start[0], 
			param->need_scaling?"yes":"no") ;
	fprintf(fp, "\n\nIter\ttime\trms_change\tinfo_rate\tlog-likelihood\tnum_rot\tbeta\n") ;
	fclose(fp) ;
}

void update_log_file(double iter_time, double likelihood, double beta) {
	FILE *fp = fopen(param->log_fname, "a") ;
	fprintf(fp, "%d\t", param->iteration) ;
	fprintf(fp, "%4.2f\t", iter_time) ;
	fprintf(fp, "%1.4e\t%f\t%.6e\t%-7d\t%f\n", iter->rms_change, iter->mutual_info, likelihood, quat->num_rot, beta) ;
	fclose(fp) ;
}

void save_models() {
#ifndef WITH_HDF5
	FILE *fp ;
	char fname[2048] ;
	
	sprintf(fname, "%s/output/intens_%.3d.bin", param->output_folder, param->iteration) ;
	fp = fopen(fname, "w") ;
	fwrite(iter->model1, sizeof(double), iter->modes * iter->vol, fp) ;
	fclose(fp) ;
	
	sprintf(fname, "%s/weights/weights_%.3d.bin", param->output_folder, param->iteration) ;
	fp = fopen(fname, "w") ;
	fwrite(iter->inter_weight, sizeof(double), iter->modes * iter->vol, fp) ;
	fclose(fp) ;

	// Write scale factors to file even when not updating them
	if (param->need_scaling) {	
		char fname[2048] ;
		sprintf(fname, "%s/scale/scale_%.3d.dat", param->output_folder, param->iteration) ;
		FILE *fp_scale = fopen(fname, "w") ;
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			fprintf(fp_scale, "%.15e\n", iter->scale[d]) ;
		fclose(fp_scale) ;
	}
	
#else // WITH_HDF5

	hid_t file, dset, dspace ;
	char name[2048] ;
	hsize_t out_size3d[4], out_size2d[3] ;
	out_size3d[0] = iter->modes ;
	out_size3d[1] = iter->size ;
	out_size3d[2] = iter->size ;
	out_size3d[3] = iter->size ;
	out_size2d[0] = iter->modes ;
	out_size2d[1] = iter->size ;
	out_size2d[2] = iter->size ;
	
	sprintf(name, "%s/output_%.3d.h5", param->output_folder, param->iteration) ;
	//file = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT) ;
	file = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) ;
	
	if (param->recon_type == RECON2D || param->recon_type == RECONRZ)
		dspace = H5Screate_simple(3, out_size2d, NULL) ;
	else
		dspace = H5Screate_simple(4, out_size3d, NULL) ;
	dset = H5Dcreate(file, "/intens", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
	H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, iter->model1) ;
	H5Dclose(dset) ;
	
	dset = H5Dcreate(file, "/inter_weight", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
	H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, iter->inter_weight) ;
	H5Sclose(dspace) ;
	H5Dclose(dset) ;
	
	if (param->need_scaling) {
		hsize_t len[1] ;
		len[0] = frames->tot_num_data ;
		dspace = H5Screate_simple(1, len, NULL) ;
		dset = H5Dcreate(file, "scale", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
		H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, iter->scale) ;
		H5Sclose(dspace) ;
		H5Dclose(dset) ;
	}
	
	H5Fclose(file) ;
#endif //WITH_HDF5
}

double maximize() {
	double *info = calloc(frames->tot_num_data, sizeof(double)) ;
	double *likelihood = calloc(frames->tot_num_data, sizeof(double)) ;
	memset(iter->model2, 0, iter->modes * iter->vol * sizeof(double)) ;
	memset(iter->inter_weight, 0, iter->modes * iter->vol * sizeof(double)) ;
	
	#pragma omp parallel default(shared)
	{
		int omp_rank = omp_get_thread_num() ;
		int cd, start_d, curr_d, d, m, t, pixel, old_r = -2 ;
		double temp, max_exp, p_sum ;
		struct dataset *curr = frames ;
		double *view, *views = malloc(param->modes * det->num_pix * sizeof(double)) ;
		double *mergeview = malloc(det->num_pix * sizeof(double)) ;
		double *priv_model = calloc(iter->vol * iter->modes, sizeof(double)) ;
		double *priv_weight = calloc(iter->vol * iter->modes, sizeof(double)) ;
		double *prob = malloc(param->modes * sizeof(double)) ;
		
		for (d = 0 ; d < iter->tot_num_data ; ++d)
		if (rots[d] > -1)
			break ;
		start_d = d ;
		
		while (curr != NULL) {
			for (cd = 0 ; cd < curr->tot_num_data ; ++cd)
			if (rots[sort_args[curr->num_data_prev + cd]] > -1)
				break ;
			start_d = cd - 1 ;
			if (omp_rank == 0)
				fprintf(stderr, "%s: start_d = %d\n", curr->filename, start_d) ;
			
			#pragma omp for schedule(static)
			//for (cd = start_d ; cd < start_d+5000 ; ++cd) {
			for (cd = start_d ; cd < curr->num_data ; ++cd) {
				// Sorted frame number
				d = sort_args[curr->num_data_prev + cd] ;
				
				// Sorted frame number in current file
				curr_d = d - curr->num_data_prev ;
				
				// Check if frame is blacklisted
				if (frames->blacklist[d])
					continue ;
				
				// Update views if orientation has changed
				if (rots[d] != old_r) {
					for (m = 0 ; m < param->modes ; ++m)
						slice_gen3d(&quat->quat[5*rots[d]], 1., &views[m*det->num_pix], &(iter->model1[m*iter->vol]), iter->size, det) ;
					old_r = rots[d] ;
				}
				
				// Calculate log expectation for each mode
				max_exp = -1e20 ;
				p_sum = 0. ;
				for (m = 0 ; m < param->modes ; ++m) {
					// Initialize probabilities
					if (param->need_scaling)
						prob[m] = iter->scale[d] ;
					else
						//prob[m] = iter->rescale ;
						prob[m] = 0. ;
					
					view = &views[m*det->num_pix] ;
					
					// For each pixel with one photon
					for (t = 0 ; t < curr->ones[curr_d] ; ++t) {
						pixel = curr->place_ones[curr->ones_accum[curr_d] + t] ;
						if (det->mask[pixel] < 1)
							prob[m] += view[pixel] ;
					}
					
					// For each pixel with count_multi photons
					for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
						pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
						if (det->mask[pixel] < 1)
							prob[m] += curr->count_multi[curr->multi_accum[curr_d] + t] * view[pixel] ;
					}
					
					if (prob[m] > max_exp)
						max_exp = prob[m] ;
				}
				
				// Exponentiate and normalize probabilities
				for (m = 0 ; m < param->modes ; ++m) {
					p_sum += exp(param->beta[d] * (prob[m] - max_exp)) ;
				}
				
				// Calculate updated tomograms and merge
				for (m = 0 ; m < param->modes ; ++m) {
					temp = prob[m] ;
					prob[m] = exp(param->beta[d] * (prob[m] - max_exp)) / p_sum ;
					
					if (prob[m] < 1e-4)
						continue ;
					
					if (param->need_scaling)
						likelihood[d] += prob[m] * (temp - frames->sum_fact[d] + frames->count[d]*log(iter->scale[d])) ;
					else
						likelihood[d] += prob[m] * (temp - frames->sum_fact[d]) ;
					
					info[d] += prob[m] * log(prob[m] * iter->modes) ;

					memset(mergeview, 0, sizeof(double)*det->num_pix) ;
					
					// For each pixel with one photon
					for (t = 0 ; t < curr->ones[curr_d] ; ++t) {
						pixel = curr->place_ones[curr->ones_accum[curr_d] + t] ;
						if (det->mask[pixel] < 2)
							mergeview[pixel] = prob[m] / iter->scale[d] ;
					}
					
					// For each pixel with count_multi photons
					for (t = 0 ; t < curr->multi[curr_d] ; ++t) {
						pixel = curr->place_multi[curr->multi_accum[curr_d] + t] ;
						if (det->mask[pixel] < 2)
							mergeview[pixel] += curr->count_multi[curr->multi_accum[curr_d] + t] * prob[m] / iter->scale[d] ;
					}
					
					slice_merge3d(&quat->quat[5*rots[d]], mergeview, &priv_model[m*iter->vol], &priv_weight[m*iter->vol], iter->size, det) ;
				}
				
				if (omp_rank == 0 && cd%100 == 0)
					fprintf(stderr, "\r%.5d/%.5d: %.5d %.5d", cd, curr->num_data, d, rots[d]) ;
			}
			if (omp_rank == 0)
				fprintf(stderr, "\n") ;
			
			curr = curr->next ;
		}
		
		#pragma omp critical(model)
		{
			long x ;
			for (x = 0 ; x < iter->vol * iter->modes ; ++x) {
				iter->model2[x] += priv_model[x] ;
				iter->inter_weight[x] += priv_weight[x] ;
			}
		}
		free(views) ;
		free(prob) ;
		free(priv_model) ;
		free(priv_weight) ;
	}
	
	int d ;
	double avg_likelihood = 0 ;
	iter->mutual_info = 0. ;
	for (d = 0 ; d < frames->tot_num_data ; ++d) {
		iter->mutual_info += info[d] ;
		avg_likelihood += likelihood[d] ;
	}
	iter->mutual_info /= (frames->tot_num_data - frames->num_blacklist) ;
	avg_likelihood /= (frames->tot_num_data - frames->num_blacklist) ;
	
	free(info) ;
	free(likelihood) ;
	
	return avg_likelihood ;
}

double update_beta() {
	int d ;
	double beta_mean = 0. ;
	
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (!frames->blacklist[d]) {
		param->beta[d] = param->beta_start[d] * pow(param->beta_jump, (param->iteration-1) / param->beta_period) ;
		if (param->beta[d] > 1.)
			param->beta[d] = 1. ;
		beta_mean += param->beta[d] ;
	}
	beta_mean /= (frames->tot_num_data - frames->num_blacklist) ;
	
	return beta_mean ;
}

void update_model(double likelihood) {
	long x ;
	double diff, change = 0. ;
	
	for (x = 0 ; x < iter->modes * iter->vol ; ++x)
	if (iter->inter_weight[x] > 0.)
		iter->model2[x] /= iter->inter_weight[x] ;
	
	for (x = 0 ; x < param->modes ; ++x)
		symmetrize_friedel(&iter->model2[x*iter->vol], iter->size) ;
	
	for (x = 0 ; x < iter->modes * iter->vol ; ++x) {
		diff = iter->model2[x] - iter->model1[x] ;
		change += diff * diff ;
		if (param->alpha > 0.)
			iter->model1[x] = param->alpha * iter->model1[x] + (1. - param->alpha) * iter->model2[x] ;
		else
			iter->model1[x] = iter->model2[x] ;
	}
	iter->rms_change = sqrt(change / iter->modes / iter->vol) ;
}

void emc() {
	double likelihood, beta_mean ;
	struct timeval t1, t2, t3 ;
	
	for (param->iteration = param->start_iter ; param->iteration <= param->num_iter + param->start_iter - 1 ; ++param->iteration) {
		gettimeofday(&t1, NULL) ;
		
		beta_mean = update_beta() ;
		
		likelihood = maximize() ;
		print_recon_time("Completed maximize", &t1, &t2, 0) ;
		
		update_model(likelihood) ;
		if (param->need_scaling && param->recon_type == RECON3D)
			normalize_scale(frames, iter) ;
		if (!param->rank) {
			save_models() ;
			gettimeofday(&t2, NULL) ;
			update_log_file((double)(t2.tv_sec - t1.tv_sec) + 1.e-6*(t2.tv_usec - t1.tv_usec), likelihood, beta_mean) ;
		}
		print_recon_time("Updated 3D intensity", &t2, &t3, 0) ;
		
		if (isnan(iter->rms_change)) {
			fprintf(stderr, "rms_change = NAN\n") ;
			break ;
		}
	}
	
	fprintf(stderr, "Finished all iterations\n") ;
}

void free_memory() {
	free_iterate(iter) ;
	free_data(param->need_scaling, frames) ;
	free_quat(quat) ;
	free_detector(det) ;
	free(param) ;
}

int main(int argc, char *argv[]) {
	int num_iter, num_threads = omp_get_max_threads() ;
	char config_fname[1024] ;
	
	num_iter = parse_arguments(argc, argv, &num_threads, config_fname) ;
	if (num_iter < 0)
		return 1 ;
	
	if (setup(config_fname))
		return 1 ;
	param->num_iter = num_iter ;
	write_log_file_header(num_threads) ;
	
	emc() ;
	
	free_memory() ;
	
	return 0 ;
}
