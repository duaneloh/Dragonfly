from posix.time cimport timeval
from libc.stdint cimport uint8_t
cimport emc

cdef extern from "../src/max_emc.c" nogil:
    timeval tm1
    timeval tm2

    cdef void allocate_memory(emc.max_data*)
    cdef double calculate_rescale(emc.max_data*)
    cdef void calculate_prob(int, emc.max_data*, emc.max_data*)
    cdef void normalize_prob(emc.max_data*, emc.max_data*)
    cdef void update_tomogram(int, emc.max_data*, emc.max_data*)
    cdef void merge_tomogram(int, emc.max_data*)
    cdef void combine_information_omp(emc.max_data*, emc.max_data*)
    cdef double combine_information_mpi(emc.max_data*)
    cdef void update_scale(emc.max_data*)
    cdef void free_memory(emc.max_data*)

cdef class py_max_data:
    cdef emc.max_data* data

cdef class py_maximize:
    cdef emc.max_data* common_data
    cdef emc.max_data* priv_data

