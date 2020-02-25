cimport decl

cdef class detector:
    cdef decl.detector* det
    cdef int curr_det
    cdef int num_modes

