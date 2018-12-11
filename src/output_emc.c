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
}

void update_log_file(double iter_time, double likelihood) {
	FILE *fp = fopen(param->log_fname, "a") ;
	fprintf(fp, "%d\t", param->iteration) ;
	fprintf(fp, "%4.2f\t", iter_time) ;
	fprintf(fp, "%1.4e\t%f\t%.6e\t%-7d\t%f\n", iter->rms_change, iter->mutual_info, likelihood, quat->num_rot, param->beta) ;
	fclose(fp) ;
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
	out_size3d[0] = param->modes ;
	out_size3d[1] = iter->size ;
	out_size3d[2] = iter->size ;
	out_size3d[3] = iter->size ;
	out_size2d[0] = param->modes ;
	out_size2d[1] = iter->size ;
	out_size2d[2] = iter->size ;
	
	sprintf(name, "%s/output_000.h5", param->output_folder) ;
	file = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) ;
	
	if (param->recon_type == RECON2D)
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

void save_models() {
#ifndef WITH_HDF5
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

#else // WITH_HDF5

	hid_t file, dset, dspace ;
	char name[2048] ;
	hsize_t out_size3d[4], out_size2d[3] ;
	out_size3d[0] = param->modes ;
	out_size3d[1] = iter->size ;
	out_size3d[2] = iter->size ;
	out_size3d[3] = iter->size ;
	out_size2d[0] = param->modes ;
	out_size2d[1] = iter->size ;
	out_size2d[2] = iter->size ;
	
	sprintf(name, "%s/output_%.3d.h5", param->output_folder, param->iteration) ;
	file = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT) ;
	
	if (param->recon_type == RECON2D)
		dspace = H5Screate_simple(3, out_size2d, NULL) ;
	else
		dspace = H5Screate_simple(4, out_size3d, NULL) ;
	dset = H5Dcreate(file, "/intens", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
	H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, iter->model1) ;
	H5Dclose(dset) ;
	
	dset = H5Dcreate(file, "/inter_weight", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
	H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, iter->inter_weight) ;
	H5Dclose(dset) ;
	
	H5Sclose(dspace) ;
	H5Fclose(file) ;
#endif //WITH_HDF5
}

void save_metrics(struct max_data *data) {
#ifndef WITH_HDF5
	int d ;
	
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
	
	// Write scale factors to file even when not updating them
	if (param->need_scaling) {	
		char fname[2048] ;
		sprintf(fname, "%s/scale/scale_%.3d.dat", param->output_folder, param->iteration) ;
		FILE *fp_scale = fopen(fname, "w") ;
		for (d = 0 ; d < frames->tot_num_data ; ++d)
			fprintf(fp_scale, "%.15e\n", iter->scale[d]) ;
		fclose(fp_scale) ;
	}
	
	// Write frame-by-frame mode occupancies to file
	if (param->modes > 1) {
		sprintf(fname, "%s/modes/occupancies_%.3d.bin", param->output_folder, param->iteration) ;
		FILE *fp_modes = fopen(fname, "w") ;
		fwrite(data->quat_norm, sizeof(double), frames->tot_num_data*param->modes, fp_modes) ;
		fclose(fp_modes) ;
	}

#else // WITH_HDF5

	char name[2048] ;
	hid_t file, dset, dspace ;
	hsize_t len[1] ;
	
	len[0] = frames->tot_num_data ;
	sprintf(name, "%s/output_%.3d.h5", param->output_folder, param->iteration) ;
	file = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) ;
	
	dspace = H5Screate_simple(1, len, NULL) ;
	dset = H5Dcreate(file, "orientations", H5T_STD_I32LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
	H5Dwrite(dset, H5T_STD_I32LE, H5S_ALL, dspace, H5P_DEFAULT, data->rmax) ;
	H5Dclose(dset) ;
	
	dset = H5Dcreate(file, "mutual_info", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
	H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, data->info) ;
	H5Dclose(dset) ;
	
	dset = H5Dcreate(file, "likelihood", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
	H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, data->likelihood) ;
	H5Dclose(dset) ;
	
	if (param->need_scaling) {
		dset = H5Dcreate(file, "scale", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
		H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, iter->scale) ;
		H5Dclose(dset) ;
	}
	
	if (param->modes > 1) {
		len[0] = param->modes * frames->tot_num_data ;
		dspace = H5Screate_simple(1, len, NULL) ;
		dset = H5Dcreate(file, "occupancies", H5T_IEEE_F64LE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT) ;
		H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, dspace, H5P_DEFAULT, data->quat_norm) ;
		H5Dclose(dset) ;
	}
	
	H5Sclose(dspace) ;
	H5Fclose(file) ;
#endif //WITH_HDF5
}

