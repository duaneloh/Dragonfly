#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include "emc.h"

static struct timeval tr1, tr2, tr3 ;

static int parse_arguments(int, char**, int*, int*, char*) ;
static void emc(void) ;
static void update_model() ;
static double update_beta(void) ;
static void update_radius(struct detector*) ;
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
	//free_mem() ;
	
	MPI_Finalize() ;
	
	return 0 ;
}

static int parse_arguments(int argc, char *argv[], int *continue_flag, int *num_threads, char *config_fname) {
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

static void emc() {
	double beta_mean, likelihood ;
	
	// Keep copy of original detector
	struct detector *orig_det = calloc(1, sizeof(struct detector)) ;
	copy_detector(det, orig_det) ;
	
	// Initial radius estimate if not provided
	if (param->radius == 0 && param->radius_jump > 0)
		param->radius = 4 * param->oversampling - param->radius_jump ;
	
	for (param->iteration = param->start_iter ; param->iteration <= param->num_iter + param->start_iter - 1 ; ++param->iteration) {
		gettimeofday(&tr1, NULL) ;
		
		MPI_Bcast(iter->model1, iter->modes * iter->vol, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
		
		// Increasing beta by a factor of 'beta_jump' every 'beta_period' param->iterations or by 'beta_factor'
		beta_mean = update_beta() ;
		
		// Regenerating detector mask using radius_jump and radius_period parameters
		update_radius(orig_det) ;
		
		likelihood = maximize() ;
		print_recon_time("Completed maximize", &tr1, &tr2, param->rank) ;
		
		if (!param->rank)
			update_model() ;
		if (param->need_scaling && param->recon_type == RECON3D)
			normalize_scale(frames, iter) ;
		if (!param->rank) {
			save_models() ;
			gettimeofday(&tr2, NULL) ;
			update_log_file((double)(tr2.tv_sec - tr1.tv_sec) + 1.e-6*(tr2.tv_usec - tr1.tv_usec), likelihood, beta_mean) ;
		}
		print_recon_time("Updated 3D intensity", &tr2, &tr3, param->rank) ;
		
		MPI_Bcast(&iter->rms_change, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
		if (isnan(iter->rms_change)) {
			if (!param->rank)
				fprintf(stderr, "rms_change = NAN\n") ;
			break ;
		}
	}
	
	if (!param->rank)
		fprintf(stderr, "Finished all iterations\n") ;

	free_detector(orig_det) ;
}

static void update_model() {
	long x ;
	double diff, change = 0., norm = 1. ;
	
	for (x = 0 ; x < iter->modes * iter->vol ; ++x)
		if (iter->inter_weight[x] > 0.)
			iter->model2[x] *= norm / iter->inter_weight[x] ;
	
	if (param->recon_type == RECONRZ || (param->recon_type == RECON2D && param->friedel_sym))
		symmetrize_friedel2d(iter->model2, param->modes, iter->size) ;
	else if (param->recon_type == RECON3D && quat->icosahedral_flag)
		for (x = 0 ; x < param->modes ; ++x)
			symmetrize_icosahedral(&iter->model2[x*iter->vol], iter->size) ;
	else if (param->recon_type == RECON3D && quat->octahedral_flag)
		for (x = 0 ; x < param->modes ; ++x)
			symmetrize_octahedral(&iter->model2[x*iter->vol], iter->size) ;
	else if (param->recon_type == RECON3D)
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

static double update_beta() {
	int d ;
	double factor, beta_mean = 0. ;
	if (param->beta_factor <= 0.)
		factor = pow(param->beta_jump, (param->iteration-1) / param->beta_period) ;
	else
		factor = param->beta_factor ;
	
	for (d = 0 ; d < frames->tot_num_data ; ++d)
	if (!frames->blacklist[d]) {
		param->beta[d] = param->beta_start[d] * factor ;
		if (param->beta[d] > 1.)
			param->beta[d] = 1. ;
		beta_mean += param->beta[d] ;
	}
	beta_mean /= (frames->tot_num_data - frames->num_blacklist) ;
	
	return beta_mean ;
}

static void update_radius(struct detector *orig_det) {
	// TODO Extend to multiple detectors
	// TODO Extend to symmetry operations of quaternion
	int new_num_div ;
	
	if (param->radius_jump > 0 && param->iteration % param->radius_period == 1) {
		param->radius += param->radius_jump ;
		
		free_detector(det) ;
		det = calloc(1, sizeof(struct detector)) ;
		copy_detector(orig_det, det) ;
		remask_detector(det, param->radius) ;
		if (!param->rank)
			fprintf(stderr, "Increasing radius to %.3f (%d relevant pixels)\n", param->radius, det->rel_num_pix) ;
		
		new_num_div = ceil(param->radius / param->oversampling) ;
		free_quat(quat) ;
		quat = calloc(1, sizeof(struct rotation)) ;
		quat_gen(new_num_div, quat) ;
		divide_quat(param->rank, param->num_proc, param->modes, param->nonrot_modes, quat) ;
		if (!param->rank)
			fprintf(stderr, "New quaternion sampling level: %d (%d orientations)\n", new_num_div, quat->num_rot) ;
	}
}

static void print_recon_time(char *message, struct timeval *time_1, struct timeval *time_2, int rank) {
	if (!rank) {
		gettimeofday(time_2, NULL) ;
		fprintf(stderr, "%s: %f s\n", message, (double)(time_2->tv_sec - time_1->tv_sec) + 1.e-6*(time_2->tv_usec - time_1->tv_usec)) ;
	}
}
