cimport emc
cimport openmp
import os
from detector cimport detector
from dataset cimport dataset
from quat cimport rotation
from iterate cimport iterate
from params cimport params

det = detector()
dset = dataset()
quat = rotation()
itr = iterate()
param = params()

def setup(config_fname='config.ini', continue_flag=False):
	global det, dset, quat, itr, param
	if not os.path.exists(config_fname):
		print 'Config file not found'
		return
	config_fname = os.path.abspath(config_fname)
	param.generate_params(config_fname)
	param.generate_output_dirs()
	qmax = det.generate_detectors(config_fname)
	quat.generate_quaternion(config_fname)
	quat.divide_quat(0, 1)
	dset.generate_data(config_fname, det)
	dset.generate_blacklist(config_fname)
	itr.generate_iterate(config_fname, qmax, param, det, dset)

def main(int num_iter, continue_flag=False, num_threads=openmp.omp_get_max_threads(), config_fname='config.ini'):
	if emc.setup(config_fname, int(continue_flag)) != 0:
		print 'Error in setup()'
		return
	emc.write_log_file_header(num_threads)
	emc.free_mem()
