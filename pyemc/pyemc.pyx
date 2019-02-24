import os
import sys
import numpy as np
from mpi4py import MPI

from libc.stdlib cimport malloc, free
from libc.float cimport DBL_MAX
from libc.stdio cimport fprintf, stderr
from libc.math cimport isnan
from posix.time cimport gettimeofday 
from cython.parallel import parallel, prange
cimport openmp
cimport numpy as np

cimport decl
cimport pyemc
cimport emc
cimport max_emc
cimport recon_emc
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
    if not continue_flag:
        write_log_file_header(num_threads)
    emc(num_iter)
    free_mem()
    return 0

def setup(bytes config_fname=b'config.ini', bint continue_flag=False):
    det = detector()
    frames = dataset(det)
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
        pyemc.slice_gen = decl.slice_gen3d
        pyemc.slice_merge = decl.slice_merge3d
    else:
        pyemc.slice_gen = decl.slice_gen2d
        pyemc.slice_merge = decl.slice_merge2d
    qmax = det.generate_detectors(config_fname)
    quat.generate_quaternion(config_fname)
    quat.divide_quat(param.rank, param.num_proc, param.modes)
    frames.generate_data(config_fname)
    frames.generate_blacklist(config_fname)
    iter.generate_iterate(config_fname, qmax, param, det, frames, continue_flag=int(continue_flag))
    
    # Set global struct variables to C structs within the interface types
    pyemc.det = det.det
    pyemc.frames = frames.dset
    pyemc.iter = iter.iterate
    pyemc.quat = quat.rot
    pyemc.param = param.param
    
    return 0

def write_log_file_header(int num_threads):
    fp = open(pyemc.param.log_fname, "w")
    fp.write("Cryptotomography with the EMC algorithm using MPI+OpenMP\n\n")
    fp.write("Data parameters:\n")
    if (pyemc.frames.num_blacklist == 0):
        fp.write("\tnum_data = %d\n\tmean_count = %f\n\n" % (pyemc.frames.tot_num_data, pyemc.frames.tot_mean_count))
    else:
        fp.write("\tnum_data = %d/%d\n\tmean_count = %f\n\n" % (pyemc.frames.tot_num_data-pyemc.frames.num_blacklist, pyemc.frames.tot_num_data, pyemc.frames.tot_mean_count))
    fp.write("System size:\n")
    fp.write("\tnum_rot = %d\n\tnum_pix = %d/%d\n\tsystem_volume = %ld X %ld X %ld\n\n" % (
            pyemc.quat.num_rot, pyemc.det.rel_num_pix, pyemc.det.num_pix, 
            pyemc.param.modes if pyemc.param.modes > 0 else pyemc.iter.size, pyemc.iter.size, pyemc.iter.size))
    fp.write("Reconstruction parameters:\n")
    fp.write("\tnum_threads = %d\n\tnum_proc = %d\n\talpha = %.6f\n\tbeta = %.6f\n\tneed_scaling = %s" % (
            num_threads, pyemc.param.num_proc, pyemc.param.alpha, 
            pyemc.param.beta, "yes" if pyemc.param.need_scaling else "no"))
    fp.write("\n\nIter\ttime\trms_change\tinfo_rate\tlog-likelihood\tnum_rot\tbeta\n")
    fp.close()

def free_mem():
    decl.free_iterate(pyemc.iter)
    pyemc.iter = NULL
    decl.free_data(pyemc.param.need_scaling, pyemc.frames)
    pyemc.frames = NULL
    decl.free_quat(pyemc.quat)
    pyemc.quat = NULL
    decl.free_detector(pyemc.det)
    pyemc.det = NULL
    free(pyemc.param)
    pyemc.param = NULL

def maximize():
    gettimeofday(&max_emc.tm1, NULL)
    
    common_data = <emc.max_data*>malloc(sizeof(emc.max_data))
    common_data.within_openmp = 0
    max_emc.allocate_memory(common_data)
    
    max_emc.calculate_rescale(common_data)
    
    cdef int nrp = pyemc.quat.num_rot_p
    cdef Py_ssize_t r
    with nogil, parallel():
        #fprintf(stderr, 'Rank %d reporting\n', openmp.omp_get_thread_num())
        priv_data = <emc.max_data*>malloc(sizeof(emc.max_data))
        priv_data.within_openmp = 1
        max_emc.allocate_memory(priv_data)
        
        for r in prange(3240, schedule='static', chunksize=1):
            max_emc.calculate_prob(r, priv_data, common_data)
        max_emc.normalize_prob(priv_data, common_data)
        for r in prange(3240, schedule='static', chunksize=1):
            max_emc.update_tomogram(r, priv_data, common_data)
            max_emc.merge_tomogram(r, priv_data)
        max_emc.combine_information_omp(priv_data, common_data)
        max_emc.free_memory(priv_data)
    cdef double likelihood = max_emc.combine_information_mpi(common_data)
    if pyemc.param.need_scaling and pyemc.param.update_scale:
        max_emc.update_scale(common_data)
    if not pyemc.param.rank:
        emc.save_metrics(common_data)
        emc.save_prob(common_data)
    max_emc.free_memory(common_data)
    return likelihood

def emc(int num_iter):
    pyemc.param.num_iter = num_iter
    cdef long vol
    cdef double likelihood
    if (pyemc.param.modes > 0):
        vol = pyemc.param.modes * pyemc.iter.size * pyemc.iter.size
    else:
        vol = pyemc.iter.size * pyemc.iter.size * pyemc.iter.size
    
    for pyemc.param.iteration in range(pyemc.param.start_iter, pyemc.param.num_iter + pyemc.param.start_iter):
        gettimeofday(&recon_emc.tr1, NULL)
        
        MPI.COMM_WORLD.Bcast([<double[:vol]>pyemc.iter.model1, MPI.DOUBLE], 0)
        
        # Increasing beta by a factor of 'beta_jump' every 'beta_period' param.iterations
        if pyemc.param.iteration % pyemc.param.beta_period == 1 and pyemc.param.iteration > 1:
            pyemc.param.beta *= pyemc.param.beta_jump
        
        likelihood = maximize()
        recon_emc.print_recon_time("Completed maximize", &recon_emc.tr1, &recon_emc.tr2, pyemc.param.rank)
        
        if not pyemc.param.rank:
            update_model(likelihood)
        if pyemc.param.need_scaling and pyemc.param.modes == 0:
            decl.normalize_scale(pyemc.frames, pyemc.iter)
        if not pyemc.param.rank:
            emc.save_models()
            gettimeofday(&recon_emc.tr2, NULL)
            emc.update_log_file(<double>(recon_emc.tr2.tv_sec - recon_emc.tr1.tv_sec) + 1.e-6*(recon_emc.tr2.tv_usec - recon_emc.tr1.tv_usec), likelihood)
        recon_emc.print_recon_time("Updated 3D intensity", &recon_emc.tr2, &recon_emc.tr3, pyemc.param.rank)
        
        MPI.COMM_WORLD.Bcast([<double[:1]>&pyemc.iter.rms_change, MPI.DOUBLE], 0)
        if (isnan(pyemc.iter.rms_change)):
            fprintf(stderr, "rms_change = NaN\n")
            break
    
    if not pyemc.param.rank:
        fprintf(stderr, "Finished all iterations\n")

def update_model(double likelihood):
    cdef long x, vol
    if pyemc.param.modes > 0:
        vol = pyemc.param.modes * pyemc.iter.size * pyemc.iter.size
    else:
        vol = pyemc.iter.size * pyemc.iter.size * pyemc.iter.size
    
    model1 = np.asarray(<double[:vol]> pyemc.iter.model1)
    model2 = np.asarray(<double[:vol]> pyemc.iter.model2)
    inter_weight = np.asarray(<double[:vol]> pyemc.iter.inter_weight)
    
    model2[inter_weight > 0.] *= 1. / inter_weight[inter_weight > 0.]
    
    if pyemc.param.modes > 0:
        pass
    elif pyemc.quat.icosahedral_flag:
        decl.symmetrize_icosahedral(pyemc.iter.model2, pyemc.iter.size)
    else:
        decl.symmetrize_friedel(pyemc.iter.model2, pyemc.iter.size)
    
    pyemc.iter.rms_change = np.linalg.norm(model2 - model1)
    for x in range(vol):
        if (pyemc.param.alpha > 0.):
            pyemc.iter.model1[x] = pyemc.param.alpha * pyemc.iter.model1[x] + (1. - pyemc.param.alpha) * pyemc.iter.model2[x]
        else:
            pyemc.iter.model1[x] = pyemc.iter.model2[x]
