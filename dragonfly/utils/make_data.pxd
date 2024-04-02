from dragonfly.detector cimport detector
from dragonfly.model cimport model

cdef extern from 'src/make_data.c' nogil:
    double rescale_intens(double, double, model*, detector*)
    double gen_and_save_dataset(long, double, long, char*, long, char*, char*, model*, detector*)
