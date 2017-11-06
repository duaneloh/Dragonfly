#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include "emc.h"

int parse_arguments(int, char**, int*, int*, char*) ;
void write_log_file_header(int) ;

int main(int argc, char *argv[]) {
	int x, continue_flag = 0 ;
	int num_threads = omp_get_max_threads() ;
	long vol ;
	double change, norm, diff, likelihood ;
	struct timeval t1, t2, t3 ;
	char fname[1024], config_fname[1024] ;
	FILE *fp ;
	
	gettimeofday(&t1, NULL) ;
	MPI_Init(&argc, &argv) ;
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc) ;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
	
	strcpy(config_fname, "config.ini") ;
	system("ls > /dev/null") ;
	system("ls .. > /dev/null") ;
	
	if (parse_arguments(argc, argv, &continue_flag, &num_threads, config_fname))
		return 1 ;
	omp_set_num_threads(num_threads) ;
	
	if (setup(config_fname, continue_flag)) {
		MPI_Finalize() ;
		return 1 ;
	}
	vol = iter->size * iter->size * iter->size ;
	
	if (!rank && !continue_flag)
		write_log_file_header(num_threads) ;
	
	if (!rank) {
		gettimeofday(&t2, NULL) ;
		fprintf(stderr, "Completed setup: %f s\n", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;
	}

	for (param.iteration = param.start_iter ; param.iteration <= param.num_iter + param.start_iter - 1 ; ++param.iteration) {
		if (!isnan(iter->rms_change)) {
			gettimeofday(&t1, NULL) ;
			
			MPI_Bcast(iter->model1, vol, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
			
			// Increasing beta by a factor of 'beta_jump' every 'beta_period' param.iterations
			if (param.iteration % param.beta_period == 1 && param.iteration > 1)
				param.beta *= param.beta_jump ;
			
			likelihood = maximize() ;
			if (!rank) {
				gettimeofday(&t2, NULL) ;
				fprintf(stderr, "Completed maximize: %f s\n", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;
			}
			
			if (likelihood == DBL_MAX) {
				fprintf(stderr, "Error in maximize\n") ;
				MPI_Finalize() ;
				return 2 ;
			}
			
			if (!rank) {
				change = 0. ;
				//norm = 1. / frames->tot_mean_count ;
				norm = 1. ;
				
				for (x = 0 ; x < vol ; ++x)
					if (iter->inter_weight[x] > 0.)
						iter->model2[x] *= norm / iter->inter_weight[x] ;
				
				if (quat->icosahedral_flag)
					symmetrize_icosahedral(iter->model2, iter->size) ;
				else
					symmetrize_friedel(iter->model2, iter->size) ;
				
				for (x = 0 ; x < vol ; ++x) {
					diff = iter->model2[x] - iter->model1[x] ;
					change += diff * diff ;
					if (param.alpha > 0.)
						iter->model1[x] = param.alpha * iter->rescale * iter->model1[x] + (1. - param.alpha) * iter->model2[x] ;
					else
						iter->model1[x] = iter->model2[x] ;
				}
				
				sprintf(fname, "%s/output/intens_%.3d.bin", param.output_folder, param.iteration) ;
				fp = fopen(fname, "w") ;
				fwrite(iter->model1, sizeof(double), vol, fp) ;
				fclose(fp) ;
				
				sprintf(fname, "%s/weights/weights_%.3d.bin", param.output_folder, param.iteration) ;
				fp = fopen(fname, "w") ;
				fwrite(iter->inter_weight, sizeof(double), vol, fp) ;
				fclose(fp) ;
				
				iter->rms_change = sqrt(change / vol) ;
				
				gettimeofday(&t3, NULL) ;
				fprintf(stderr, "Finished iteration: %f s\n", (double)(t3.tv_sec - t2.tv_sec) + (t3.tv_usec - t2.tv_usec) / 1000000.) ;
				
				gettimeofday(&t2, NULL) ;
				
				fp = fopen(param.log_fname, "a") ;
				fprintf(fp, "%d\t", param.iteration) ;
				fprintf(fp, "%4.2f\t", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;
				fprintf(fp, "%1.4e\t%f\t%.6e\t%-7d\t%f\n", iter->rms_change, iter->mutual_info, likelihood, quat->num_rot, param.beta) ;
				fclose(fp) ;
			}
			
			if (param.need_scaling)
				normalize_scale(frames, iter) ;
			MPI_Bcast(&iter->rms_change, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
		}
		else
			break ;
	}
	
	if (!rank)
		fprintf(stderr, "Finished all iterations\n") ;
	
	//free_mem() ;
	
	MPI_Finalize() ;
	
	return 0 ;
}

int parse_arguments(int argc, char *argv[], int *continue_flag, int *num_threads, char *config_fname) {
	int c ;
	extern char *optarg ;
	extern int optind ;
	param.num_iter = 0 ;
	
	while (optind < argc) {
		if ((c = getopt(argc, argv, "rc:t:")) != -1) {
			switch (c) {
				case 'r':
					*continue_flag = 1 ;
					break ;
				case 't':
					*num_threads = atoi(optarg) ;
					break ;
				case 'c':
					strcpy(config_fname, optarg) ;
					break ;
			}
		}
		else {
			param.num_iter = atoi(argv[optind]) ;
			optind++ ;
		}
	}
	
	if (param.num_iter == 0) {
		fprintf(stderr, "Format: %s [-c config_fname] [-t num_threads] [-r] num_iter\n", argv[0]) ;
		fprintf(stderr, "Default: -c config.ini -t %d\n", omp_get_max_threads()) ;
		fprintf(stderr, "Missing <num_iter>\n") ;
		return 1 ;
	}
	if (!rank)
		fprintf(stderr, "Doing %d iteration(s) using %s\n", param.num_iter, config_fname) ;
	
	return 0 ;
}

void write_log_file_header(int num_threads) {
	FILE *fp = fopen(param.log_fname, "w") ;
	fprintf(fp, "Cryptotomography with the EMC algorithm using MPI+OpenMP\n\n") ;
	fprintf(fp, "Data parameters:\n") ;
	if (frames->num_blacklist == 0)
		fprintf(fp, "\tnum_data = %d\n\tmean_count = %f\n\n", frames->tot_num_data, frames->tot_mean_count) ;
	else
		fprintf(fp, "\tnum_data = %d/%d\n\tmean_count = %f\n\n", frames->tot_num_data-frames->num_blacklist, frames->tot_num_data, frames->tot_mean_count) ;
	fprintf(fp, "System size:\n") ;
	fprintf(fp, "\tnum_rot = %d\n\tnum_pix = %d/%d\n\tsystem_volume = %ld X %ld X %ld\n\n", 
			quat->num_rot, 
			det->rel_num_pix, det->num_pix, 
			iter->size, iter->size, iter->size) ;
	fprintf(fp, "Reconstruction parameters:\n") ;
	fprintf(fp, "\tnum_threads = %d\n\tnum_proc = %d\n\talpha = %.6f\n\tbeta = %.6f\n\tneed_scaling = %s", 
			num_threads, 
			num_proc, 
			param.alpha, 
			param.beta, 
			param.need_scaling?"yes":"no") ;
	fprintf(fp, "\n\nIter\ttime\trms_change\tinfo_rate\tlog-likelihood\tnum_rot\tbeta\n") ;
	fclose(fp) ;
}
