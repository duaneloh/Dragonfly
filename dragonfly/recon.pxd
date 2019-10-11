from .iterate cimport iterate
from .detector cimport detector
from .model cimport model

cdef extern from "src/maximize.h" nogil:
    double maximize(iterate*)
    void (*slice_gen)(double*, int, double*, detector*, model*)
    void (*slice_merge)(double*, int, double*, detector*, model*)

cdef class EMCRecon:
    cdef iterate *iter
