from libc.stdint cimport uint8_t
from .detector cimport detector
from .model cimport model
from .emcfile cimport dataset
from .quaternion cimport quaternion

cdef extern from "src/iterate.h" nogil:
    struct iterate:
        detector *det
        model *mod
        dataset *dset
        quaternion *quat

        # Parameters for each frame
        int tot_num_data, num_blacklist
        double *scale
        double *bgscale
        uint8_t *blacklist

        # For refinement
        int *quat_mapping
        int **rel_quat
        int *num_rel_quat
        double **rel_prob
    
        # Parameters for each detector
        int num_det # Number of unique detectors
        int *det_mapping
        double *rescale
        double *mean_count

        # Aggregate metrics
        double likelihood, mutual_info, rms_change

cdef class Iterate:
    cdef iterate *iter
