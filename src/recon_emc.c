#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include "emc.h"

static struct timeval tr1, tr2, tr3 ;
static void print_recon_time(char*, struct timeval*, struct timeval*, int) ;

int main(int argc, char *argv[]) {
	int num_iter, continue_flag = 0, num_threads = omp_get_max_threads() ;
	char config_fname[1024] ;
	
	MPI_Init(&argc, &argv) ;
	gettimeofday(&tr1, NULL) ;
	
	num_iter = parse_arguments(argc, argv, &continue_flag, &num_threads, config_fname) ;
	if (num_iter < 0) {
		MPI_Finalize() ;
		return 1 ;
	}
	
	if (setup(config_fname, continue_flag)) {
		MPI_Finalize() ;
		return 1 ;
	}
	param->num_iter = num_iter ;
	
	if (!param->rank && !continue_flag)
		write_log_file_header(num_threads) ;
	
	emc() ;
	free_mem() ;
	
	MPI_Finalize() ;
	
	return 0 ;
}

static void print_recon_time(char *message, struct timeval *time_1, struct timeval *time_2, int rank) {
	if (!rank) {
		gettimeofday(time_2, NULL) ;
		fprintf(stderr, "%s: %f s\n", message, (double)(time_2->tv_sec - time_1->tv_sec) + 1.e-6*(time_2->tv_usec - time_1->tv_usec)) ;
	}
}

int parse_arguments(int argc, char *argv[], int *continue_flag, int *num_threads, char *config_fname) {
	int c, rank, num_iter = -1 ;
	extern char *optarg ;
	extern int optind ;
	strcpy(config_fname, "config.ini") ;
	system("ls > /dev/null") ;
	system("ls .. > /dev/null") ;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
	
	while (optind < argc) {
		if ((c = getopt(argc, argv, "rc:t:")) != -1) {
			switch (c) {
				case 'r':
					*continue_flag = 1 ;
					break ;
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
		fprintf(stderr, "Format: %s [-c config_fname] [-t num_threads] [-r] num_iter\n", argv[0]) ;
		fprintf(stderr, "Default: -c config.ini -t %d\n", omp_get_max_threads()) ;
		fprintf(stderr, "Missing <num_iter>\n") ;
		return -1 ;
	}
	if (!rank)
		fprintf(stderr, "Doing %d iteration(s) using %s\n", num_iter, config_fname) ;
	
	return num_iter ;
}

void write_log_file_header(int num_threads) {
	FILE *fp = fopen(param->log_fname, "w") ;
	fprintf(fp, "Cryptotomography with the EMC algorithm using MPI+OpenMP\n\n") ;
	fprintf(fp, "Data parameters:\n") ;
	if (frames->num_blacklist == 0)
		fprintf(fp, "\tnum_data = %d\n\tmean_count = %f\n\n", frames->tot_num_data, frames->tot_mean_count) ;
	else
		fprintf(fp, "\tnum_data = %d/%d\n\tmean_count = %f\n\n", frames->tot_num_data-frames->num_blacklist, frames->tot_num_data, frames->tot_mean_count) ;
	fprintf(fp, "System size:\n") ;
	fprintf(fp, "\tnum_rot = %d\n\tnum_pix = %d/%d\n\t", quat->num_rot, det->rel_num_pix, det->num_pix) ;
	if (param->recon_type == RECON3D)
		fprintf(fp, "system_volume = %d X %ld X %ld X %ld\n\n", param->modes, iter->size, iter->size, iter->size) ;
	else if (param->recon_type == RECON2D)
		fprintf(fp, "system_volume = %d X %ld X %ld\n\n", param->modes, iter->size, iter->size) ;
	fprintf(fp, "Reconstruction parameters:\n") ;
	fprintf(fp, "\tnum_threads = %d\n\tnum_proc = %d\n\talpha = %.6f\n\tbeta = %.6f\n\tneed_scaling = %s", 
			num_threads, 
			param->num_proc, 
			param->alpha, 
			param->beta, 
			param->need_scaling?"yes":"no") ;
	fprintf(fp, "\n\nIter\ttime\trms_change\tinfo_rate\tlog-likelihood\tnum_rot\tbeta\n") ;
	fclose(fp) ;
}

void emc() {
	double likelihood ;
	
	for (param->iteration = param->start_iter ; param->iteration <= param->num_iter + param->start_iter - 1 ; ++param->iteration) {
		gettimeofday(&tr1, NULL) ;
		
		MPI_Bcast(iter->model1, param->modes * iter->vol, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
		
		// Increasing beta by a factor of 'beta_jump' every 'beta_period' param->iterations
		if (param->iteration % param->beta_period == 1 && param->iteration > 1)
			param->beta *= param->beta_jump ;
		
		likelihood = maximize() ;
		print_recon_time("Completed maximize", &tr1, &tr2, param->rank) ;
		
		if (!param->rank)
			update_model(likelihood) ;
		if (param->need_scaling && param->recon_type == RECON3D)
			normalize_scale(frames, iter) ;
		print_recon_time("Updated 3D intensity", &tr2, &tr3, param->rank) ;
		
		MPI_Bcast(&iter->rms_change, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
		if (isnan(iter->rms_change)) {
			fprintf(stderr, "rms_change = NAN\n") ;
			break ;
		}
	}
	
	if (!param->rank)
		fprintf(stderr, "Finished all iterations\n") ;
}

void update_model(double likelihood) {
	long x ;
	double diff, change = 0., norm = 1. ;
	char fname[1024] ;
	FILE *fp ;
	
	for (x = 0 ; x < param->modes * iter->vol ; ++x)
		if (iter->inter_weight[x] > 0.)
			iter->model2[x] *= norm / iter->inter_weight[x] ;
	
	if (param->recon_type == RECON2D) ;
	else if (quat->icosahedral_flag)
		for (x = 0 ; x < param->modes ; ++x)
			symmetrize_icosahedral(&iter->model2[x*iter->vol], iter->size) ;
	else
		for (x = 0 ; x < param->modes ; ++x)
			symmetrize_friedel(&iter->model2[x*iter->vol], iter->size) ;
	
	for (x = 0 ; x < param->modes * iter->vol ; ++x) {
		diff = iter->model2[x] - iter->model1[x] ;
		change += diff * diff ;
		if (param->alpha > 0.)
			iter->model1[x] = param->alpha * iter->rescale * iter->model1[x] + (1. - param->alpha) * iter->model2[x] ;
		else
			iter->model1[x] = iter->model2[x] ;
	}
	iter->rms_change = sqrt(change / param->modes / iter->vol) ;
	
	sprintf(fname, "%s/output/intens_%.3d.bin", param->output_folder, param->iteration) ;
	fp = fopen(fname, "w") ;
	fwrite(iter->model1, sizeof(double), param->modes * iter->vol, fp) ;
	fclose(fp) ;
	
	sprintf(fname, "%s/weights/weights_%.3d.bin", param->output_folder, param->iteration) ;
	fp = fopen(fname, "w") ;
	fwrite(iter->inter_weight, sizeof(double), param->modes * iter->vol, fp) ;
	fclose(fp) ;
	
	gettimeofday(&tr2, NULL) ;
	
	fp = fopen(param->log_fname, "a") ;
	fprintf(fp, "%d\t", param->iteration) ;
	fprintf(fp, "%4.2f\t", (double)(tr2.tv_sec - tr1.tv_sec) + 1.e-6*(tr2.tv_usec - tr1.tv_usec)) ;
	fprintf(fp, "%1.4e\t%f\t%.6e\t%-7d\t%f\n", iter->rms_change, iter->mutual_info, likelihood, quat->num_rot, param->beta) ;
	fclose(fp) ;
}
