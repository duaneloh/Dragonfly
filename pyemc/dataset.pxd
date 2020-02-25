cimport decl 
from detector cimport detector

cdef class dataset:
    cdef decl.dataset* dset
    cdef public detector det
