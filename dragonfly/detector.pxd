from libc.stdint cimport uint8_t

cdef extern from "src/detector.h" nogil:
    struct detector:
        char *fname
        int num_pix
        double *qvals
        double *corr
        uint8_t *raw_mask

        double detd, ewald_rad

        # Background
        uint8_t with_bg
        double *background

cdef class CDetector:
    cdef detector* det
