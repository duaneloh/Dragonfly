cimport emc
cimport openmp
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import os
from detector cimport detector
from dataset cimport dataset
from quat cimport rotation
from iterate cimport iterate
from params cimport params

cdef extern from "../src/emc.h":
	emc.detector *det ;
	emc.rotation *quat ;
	emc.dataset *frames ;
	emc.dataset *merge_frames ;
	emc.iterate *iter ;
	emc.params *param ;

def setup(config_fname='config.ini', continue_flag=False):
	det = detector()
	dset = dataset(det)
	merge_dset = dataset(det)
	quat = rotation()
	itr = iterate()
	param = params()

	if not os.path.exists(config_fname):
		print 'Config file not found'
		return 1
	config_fname = os.path.abspath(config_fname)
	param.generate_params(config_fname)
	param.generate_output_dirs()
	qmax = det.generate_detectors(config_fname)
	if qmax < 0: return 1
	quat.generate_quaternion(config_fname)
	quat.divide_quat(0, 1)
	dset.generate_data(config_fname)
	merge_dset.generate_data(config_fname, type_string='merge')
	dset.generate_blacklist(config_fname)
	itr.generate_iterate(config_fname, qmax, param, det, dset)
	return 0

def main(int num_iter, continue_flag=False, num_threads=openmp.omp_get_max_threads(), config_fname='config.ini'):
	global det, quat, frames, merge_frames, iter, param
	
	param = <emc.params*> PyMem_Malloc(sizeof(emc.params))
	param.rank = 0
	param.num_proc = 1
	param.num_iter = num_iter
	cdef char* c_config_fname = config_fname
	
	if emc.setup(c_config_fname, int(continue_flag)) != 0:
		print 'Error in setup()'
		return 1
	if not continue_flag:
		emc.write_log_file_header(num_threads)
	emc.emc()
	emc.free_mem()
	return 0
