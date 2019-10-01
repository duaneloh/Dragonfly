from libc.stdint cimport uint8_t

cdef struct detector:
    char *fname
    int num_pix
    double *qvals
    double *corr
    uint8_t *raw_mask
    double *background

    # For python interface
    double detd, ewald_rad
    double *cx
    double *cy
    int *x
    int *y
    uint8_t *mask
    uint8_t *assembled_mask

cdef class Detector:
    cdef detector* det
