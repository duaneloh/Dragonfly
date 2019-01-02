from posix.time cimport timeval

cdef extern from '../src/recon_emc.c' nogil:
	timeval tr1
	timeval tr2
	timeval tr3
	void print_recon_time(char*, timeval*, timeval*, int)

