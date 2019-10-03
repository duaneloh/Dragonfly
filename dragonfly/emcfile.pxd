from libc.stdint cimport uint8_t
from .detector cimport detector

cdef struct dataset:
    char *fname
    int num_data
    detector *det
    dataset *next
    
    # Sparse data
    int *ones
    int *multi
    int *place_ones
    int *place_multi
    int *count_multi
    long ones_total, multi_total
    long *ones_accum
    long *multi_accum
    
    # Dense data
    int *int_frames
    double *frames
    
cdef class CDataset:
    cdef dataset *dset
