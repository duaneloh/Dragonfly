#include "emc.h"

void sym_intens(double*, int, int) ;

int main(int argc, char *argv[]) {
	int x, continue_flag = 0 ;
	double change, norm, diff, likelihood ;
	struct timeval t1, t2, t3 ;
	char fname[500] ;
	FILE *fp ;
	
	MPI_Init(&argc, &argv) ;
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc) ;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
	
	if (argc < 3) {
		fprintf(stderr, "Format: %s <iter> <config_fname>\n", argv[0]) ;
		MPI_Finalize() ;
		return 1 ;
	}
	num_iter = atoi(argv[1]) ;
	
	if (argc > 3)
		omp_set_num_threads(atoi(argv[3])) ;
	else
		omp_set_num_threads(omp_get_max_threads()) ;
	
	
	gettimeofday(&t1, NULL) ;
	
	if (setup(argv[2])) {
		MPI_Finalize() ;
		return 1 ;
	}

	if (argc > 3) {
		continue_flag = 1 ;
		
		fp = fopen(log_fname, "r") ;
		if (fp == NULL) {
			fprintf(stderr, "No log file found to continue run\n") ;
			continue_flag = 0 ;
		}
		else {
			while (!feof(fp))
				fgets(fname, 500, fp) ;
			sscanf(fname, "%d", &iteration) ;
			fclose(fp) ;
			
			iteration += 1 ;
			fprintf(stderr, "Continuing from previous run starting from iteration %d.\n", iteration) ;
		}
	}
	
	if (!rank && !continue_flag) {
		fp = fopen(log_fname, "w") ;
		fprintf(fp, "Cryptotomography with the EMC algorithm using MPI+OpenMP\n\n") ;
		fprintf(fp, "Data parameters:\n") ;
		fprintf(fp, "\tnum_data = %d\n\tmean_count = %f\n\n", 
		        tot_num_data, 
		        tot_mean_count) ;
		fprintf(fp, "System size:\n") ;
		fprintf(fp, "\tnum_rot = %d\n\tnum_pix = %d/%d\n\tsystem_volume = %d X %d X %d\n\n", 
		        num_rot, 
		        rel_num_pix, num_pix, 
		        size, size, size) ;
		fprintf(fp, "Reconstruction parameters:\n") ;
		fprintf(fp, "\tnum_threads = %d\n\tnum_proc = %d\n\talpha = %.2f\n\tbeta = %.2f\n\tneed_scaling = %s", 
		        argc>3?atoi(argv[3]):omp_get_max_threads(), 
		        num_proc, 
		        alpha, 
		        beta, 
		        need_scaling?"yes":"no") ;
		fprintf(fp, "\n\nIteration\titer_time\trms_change\tinfo_rate\tlog-likelihood\n") ;
		fclose(fp) ;
	}
	
	if (!rank) {
		gettimeofday(&t2, NULL) ;
		fprintf(stderr, "Completed setup: %f s\n", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;
	}
	
	for (iteration = 1 ; iteration <= num_iter ; ++iteration) {
		if (!isnan(rms_change)) {
			gettimeofday(&t1, NULL) ;
			
			MPI_Bcast(model1, size*size*size, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
			
			likelihood = maximize() ;
			if (!rank) {
				gettimeofday(&t2, NULL) ;
				fprintf(stderr, "Completed maximize: %f s\n", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;
			}
			
			if (likelihood == 0.) {
				fprintf(stderr, "Error in maximize\n") ;
				MPI_Finalize() ;
				return 2 ;
			}
			
			if (rank) {
				MPI_Reduce(model2, model2, size*size*size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
				MPI_Reduce(inter_weight, inter_weight, size*size*size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
			}
			else {
				MPI_Reduce(MPI_IN_PLACE, model2, size*size*size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
				MPI_Reduce(MPI_IN_PLACE, inter_weight, size*size*size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) ;
			}
			
			if (!rank) {
				change = 0. ;
				//norm = 1. / tot_mean_count ;
				norm = 1. ;
				
				for (x = 0 ; x < size*size*size ; ++x)
					if (inter_weight[x] > 0.)
						model2[x] *= norm / inter_weight[x] ;
				
				sym_intens(model2, size, center) ;
				
				for (x = 0 ; x < size*size*size ; ++x) {
					diff = model2[x] - model1[x] ;
					change += diff * diff ;
					model1[x] = model2[x] ;
				}
				
				sprintf(fname, "%s/output/intens_%.3d.bin", output_folder, iteration) ;
				fp = fopen(fname, "w") ;
				fwrite(model1, sizeof(double), size * size * size, fp) ;
				fclose(fp) ;
				
				sprintf(fname, "%s/weights/weights_%.3d.bin", output_folder, iteration) ;
				fp = fopen(fname, "w") ;
				fwrite(inter_weight, sizeof(double), size * size * size, fp) ;
				fclose(fp) ;
				
				rms_change = sqrt(change / size / size / size) ;
				
				gettimeofday(&t3, NULL) ;
				fprintf(stderr, "Finished iteration: %f s\n", (double)(t3.tv_sec - t2.tv_sec) + (t3.tv_usec - t2.tv_usec) / 1000000.) ;
				
				gettimeofday(&t2, NULL) ;
				
				fp = fopen(log_fname, "a") ;
				fprintf(fp, "%d\t\t", iteration) ;
				fprintf(fp, "%.2f s  \t", (double)(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.) ;
				fprintf(fp, "%1.4e\t%f\t%.6e\n", rms_change, info, likelihood) ;
				fclose(fp) ;
			}
			
/*			// Rescaling model and scale factors
			double mean_scale = 0. ;
			long d ;
			for (d = 0 ; d < tot_num_data ; ++d)
				mean_scale += scale[d] ;
			mean_scale /= tot_num_data ;
			for (x = 0 ; x < size*size*size ; ++x)
				model1[x] *= mean_scale ;
			for (d = 0 ; d < tot_num_data ; ++d)
				scale[d] /= mean_scale ;
*/				
			MPI_Bcast(&rms_change, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
		}
		else
			break ;
	}
	
	if (!rank)
		fprintf(stderr, "Finished all iterations\n") ;
	
//	free_mem() ;
	
	MPI_Finalize() ;
	
	return 0 ;
}


void sym_intens(double *array, int size, int center) {
	int x, y, z ;
	double ave_intens ;
	
	for (x = 0 ; x < size ; ++x)
	for (y = 0 ; y < size ; ++y)
	for (z = 0 ; z <= center ; ++z) {
		ave_intens = .5 * (array[x*size*size + y*size + z] + array[(2*center-x)*size*size + (2*center-y)*size + (2*center-z)]) ;
		array[x*size*size + y*size + z] = ave_intens ;
		array[(2*center-x)*size*size + (2*center-y)*size +  (2*center-z)] = ave_intens ;
	}
}

