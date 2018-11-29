#include "emc.h"

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

	if (param->hdf5_out) {
		char fname[2048] ;
		hid_t file, dspace, dcpl ;
		hsize_t init_size[1] = {0} ;
		hsize_t max_size[1] = {H5S_UNLIMITED} ;
		hsize_t chunk_size[1] = {1} ;
		
		sprintf(fname, "%s.h5", param->log_fname) ;
		file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) ;
		H5Gcreate(file, "metrics", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
		H5Gcreate(file, "params", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
		dspace = H5Screate_simple(1, init_size, max_size) ;
		dcpl = H5Pcreate(H5P_DATASET_CREATE) ;
		H5Pset_chunk(dcpl, 1, chunk_size) ;
		H5Dcreate(file, "/metrics/iter_time", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT) ;
		H5Dcreate(file, "/metrics/rms_change", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT) ;
		H5Dcreate(file, "/metrics/mutual_info", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT) ;
		H5Dcreate(file, "/metrics/likelihood", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT) ;
		H5Dcreate(file, "/params/num_rot", H5T_STD_I32LE, dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT) ;
		H5Dcreate(file, "/params/beta", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT) ;
		H5Fclose(file) ;
	}
}

void update_log_file(double iter_time, double likelihood) {
	FILE *fp = fopen(param->log_fname, "a") ;
	fprintf(fp, "%d\t", param->iteration) ;
	fprintf(fp, "%4.2f\t", iter_time) ;
	fprintf(fp, "%1.4e\t%f\t%.6e\t%-7d\t%f\n", iter->rms_change, iter->mutual_info, likelihood, quat->num_rot, param->beta) ;
	fclose(fp) ;

	if (param->hdf5_out) {
		hid_t file, dset, dspace, wspace ;
		char fname[2048] ;
		hsize_t new_size[1] = {param->iteration} ;
		hsize_t sel_start[1] = {param->iteration - 1} ;
		hsize_t sel_count[1] = {1} ;
		
		sprintf(fname, "%s.h5", param->log_fname) ;
		file = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT) ;
		
		dset = H5Dopen(file, "/metrics/iter_time", H5P_DEFAULT) ;
		H5Dset_extent(dset, new_size) ;
		dspace = H5Dget_space(dset) ;
		H5Sselect_hyperslab(dspace, H5S_SELECT_SET, sel_start, NULL, sel_count, NULL) ;
		wspace = H5Screate_simple(1, sel_count, NULL) ;
		H5Dwrite(dset, H5T_IEEE_F64LE, wspace, dspace, H5P_DEFAULT, &iter_time) ;
		
		dset = H5Dopen(file, "/metrics/rms_change", H5P_DEFAULT) ;
		H5Dset_extent(dset, new_size) ;
		H5Dwrite(dset, H5T_IEEE_F64LE, wspace, dspace, H5P_DEFAULT, &(iter->rms_change)) ;
		
		dset = H5Dopen(file, "/metrics/mutual_info", H5P_DEFAULT) ;
		H5Dset_extent(dset, new_size) ;
		H5Dwrite(dset, H5T_IEEE_F64LE, wspace, dspace, H5P_DEFAULT, &(iter->mutual_info)) ;
		
		dset = H5Dopen(file, "/metrics/likelihood", H5P_DEFAULT) ;
		H5Dset_extent(dset, new_size) ;
		H5Dwrite(dset, H5T_IEEE_F64LE, wspace, dspace, H5P_DEFAULT, &likelihood) ;
		
		dset = H5Dopen(file, "/params/num_rot", H5P_DEFAULT) ;
		H5Dset_extent(dset, new_size) ;
		H5Dwrite(dset, H5T_STD_I32LE, wspace, dspace, H5P_DEFAULT, &(quat->num_rot)) ;
		
		dset = H5Dopen(file, "/params/beta", H5P_DEFAULT) ;
		H5Dset_extent(dset, new_size) ;
		H5Dwrite(dset, H5T_IEEE_F64LE, wspace, dspace, H5P_DEFAULT, &(param->beta)) ;
		
		H5Fclose(file) ;
	}
}

void save_models() {
	FILE *fp ;
	char fname[2048] ;
	
	sprintf(fname, "%s/output/intens_%.3d.bin", param->output_folder, param->iteration) ;
	fp = fopen(fname, "w") ;
	fwrite(iter->model1, sizeof(double), param->modes * iter->vol, fp) ;
	fclose(fp) ;
	
	sprintf(fname, "%s/weights/weights_%.3d.bin", param->output_folder, param->iteration) ;
	fp = fopen(fname, "w") ;
	fwrite(iter->inter_weight, sizeof(double), param->modes * iter->vol, fp) ;
	fclose(fp) ;
}

void save_metrics(struct max_data *data) {
	int d ;
	
	// Write scale factors to file even when not updating them
	if (param->need_scaling) {	
		char fname[2048] ;
		sprintf(fname, "%s/scale/scale_%.3d.dat", param->output_folder, param->iteration) ;
		FILE *fp_scale = fopen(fname, "w") ;
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			fprintf(fp_scale, "%.15e\n", iter->scale[d]) ;
		fclose(fp_scale) ;
	}
	
	/*
	if (param->modes > 1) {
		int r ;
		fprintf(stderr, "Mode occupancies: ") ;
		for (r = 0 ; r < param->modes ; ++r)
			fprintf(stderr, "%.3f ", data->quat_norm[r]/(frames->tot_num_data - frames->num_blacklist)) ;
		fprintf(stderr, "\n") ;
		for (r = 0 ; r < quat->num_rot ; ++r)
			quat->quat[r*5 + 4] = data->quat_norm[r/param->rot_per_mode] / (frames->tot_num_data - frames->num_blacklist) ;
	}
	*/

	// Print frame-by-frame mutual information, likelihood, and most likely orientations to file
	char fname[2048] ;
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
	
	// Write frame-by-frame mode occupancies to file
	if (param->modes > 1) {
		sprintf(fname, "%s/modes/occupancies_%.3d.bin", param->output_folder, param->iteration) ;
		FILE *fp_modes = fopen(fname, "w") ;
		fwrite(data->quat_norm, sizeof(double), frames->tot_num_data*param->modes, fp_modes) ;
		fclose(fp_modes) ;
	}
}

