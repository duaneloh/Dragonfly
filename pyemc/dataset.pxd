cimport emc 
from detector cimport detector

cdef class dataset:
	cdef emc.dataset* dset
	cdef public detector det
