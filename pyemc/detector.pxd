cimport emc

cdef class detector:
	cdef emc.detector* det
	cdef int curr_det
	cdef int num_modes

