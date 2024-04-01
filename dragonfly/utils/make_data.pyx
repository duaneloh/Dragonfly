import sys
import os.path as op

import numpy as np
import dragonfly
from cython.parallel import parallel, prange
cimport openmp

from .py_src import read_config

def rand_quat():
    while True:
        qvals = np.random.random(4) - 0.5
        qnorm = np.linalg.norm(qvals)
        if qnorm < 0.25:
            break
    return qvals / qnorm

def main():
    config = read_config.MyConfigParser()
    config.read('config.ini')
    
    num_data = config.getint('make_data', 'num_data')
    fluence = config.getfloat('make_data', 'fluence')
    intens_fname = config.get_filename('make_data', 'in_intensity_file')
    det_fname = config.get_filename('make_data', 'in_detector_file')
    out_fname = config.get_filename('make_data', 'out_photons_file')
    
    model = dragonfly.Model(1)
    try:
        model.allocate(intens_fname)
    except ValueError as err:
        size = err.args[1]
        model.free()
        model = dragonfly.Model(size)
        model.allocate(intens_fname)
    
    det = dragonfly.CDetector(det_fname, norm=False)

    rescale = fluence * 2.81794e-9**2
    model.model1[0] *= rescale

    hdf5_output = True
    if op.splitext(out_fname)[1] == '.emc':
        hdf5_output = False

    wemc = dragonfly.EMCWriter(out_fname, det.num_pix, hdf5=hdf5_output)
    view = np.zeros(det.num_pix, dtype='f8')
    #print(openmp.omp_get_max_threads())

    for i in range(num_data):
        phot = np.random.poisson(model.slice_gen(rand_quat(), det, view=view))
        phot[det.raw_mask == 2] = 0
        wemc.write_frame(phot.ravel())
        sys.stderr.write('\r%d/%d'%(i+1, num_data))
    sys.stderr.write('\n')

    wemc.finish_write()

if __name__ == '__main__':
    main()
