import os

cimport openmp
from libc.stdlib cimport malloc, free
from mpi4py import MPI

cimport decl
from detector cimport detector
from dataset cimport dataset
from quat cimport rotation
from iterate cimport iterate
from params cimport params

def main(int num_iter, bint continue_flag=False, int num_threads=openmp.omp_get_max_threads(), bytes config_fname=b'config.ini'):
	cdef char* c_config_fname = config_fname
	openmp.omp_set_num_threads(num_threads)

	if setup(config_fname, continue_flag) != 0:
		print 'Error in setup()'
		return 1
	decl.param.num_iter = num_iter
	if not continue_flag:
		write_log_file_header(num_threads)
	decl.emc()
	free_mem()
	return 0

cpdef setup(bytes config_fname=b'config.ini', bint continue_flag=False):
	# Cython wrapped interface types
	det = detector()
	frames = dataset(det)
	merge_frames = dataset(det)
	quat = rotation()
	iter = iterate()
	param = params()

	# Do setup using class methods
	param.param.rank = MPI.COMM_WORLD.rank
	param.param.num_proc = MPI.COMM_WORLD.size

	if not os.path.exists(config_fname):
		print 'Config file not found'
		return 1
	config_fname = os.path.abspath(config_fname)
	param.generate_params(config_fname)
	param.generate_output_dirs()
	if param.modes == 0:
		decl.slice_gen = decl.slice_gen3d
		decl.slice_merge = decl.slice_merge3d
	else:
		decl.slice_gen = decl.slice_gen2d
		decl.slice_merge = decl.slice_merge2d
	qmax = det.generate_detectors(config_fname)
	quat.generate_quaternion(config_fname)
	quat.divide_quat(param.rank, param.num_proc)
	frames.generate_data(config_fname)
	merge_frames.generate_data(config_fname, type_string='merge')
	frames.generate_blacklist(config_fname)
	iter.generate_iterate(config_fname, qmax, param, det, frames, continue_flag=int(continue_flag))

	# Set global struct variables to C structs within the interface types
	decl.det = det.det
	decl.frames = frames.dset
	decl.iter = iter.iterate
	decl.quat = quat.rot
	decl.param = param.param
	decl.merge_frames = NULL # TODO fix if merge_photons_file is provided
	
	return 0

def write_log_file_header(int num_threads):
	fp = open(decl.param.log_fname, "w")
	fp.write("Cryptotomography with the EMC algorithm using MPI+OpenMP\n\n")
	fp.write("Data parameters:\n")
	if (decl.frames.num_blacklist == 0):
		fp.write("\tnum_data = %d\n\tmean_count = %f\n\n" % (decl.frames.tot_num_data, decl.frames.tot_mean_count))
	else:
		fp.write("\tnum_data = %d/%d\n\tmean_count = %f\n\n" % (decl.frames.tot_num_data-decl.frames.num_blacklist, decl.frames.tot_num_data, decl.frames.tot_mean_count))
	fp.write("System size:\n")
	fp.write("\tnum_rot = %d\n\tnum_pix = %d/%d\n\tsystem_volume = %ld X %ld X %ld\n\n" % (
			decl.quat.num_rot, decl.det.rel_num_pix, decl.det.num_pix, 
			decl.param.modes if decl.param.modes > 0 else decl.iter.size, decl.iter.size, decl.iter.size))
	fp.write("Reconstruction decl.parameters:\n")
	fp.write("\tnum_threads = %d\n\tnum_proc = %d\n\talpha = %.6f\n\tbeta = %.6f\n\tneed_scaling = %s" % (
			num_threads, decl.param.num_proc, decl.param.alpha, 
			decl.param.beta, "yes" if decl.param.need_scaling else "no"))
	fp.write("\n\nIter\ttime\trms_change\tinfo_rate\tlog-likelihood\tnum_rot\tbeta\n")
	fp.close()

def free_mem():
	decl.free_iterate(decl.iter)
	decl.iter = NULL
	if decl.merge_frames != NULL:
		decl.free_data(decl.param.need_scaling, decl.merge_frames)
		decl.merge_frames = NULL
	decl.free_data(decl.param.need_scaling, decl.frames)
	decl.frames = NULL
	decl.free_quat(decl.quat)
	decl.quat = NULL
	decl.free_detector(decl.det)
	decl.det = NULL
	free(decl.param)
	decl.param = NULL
