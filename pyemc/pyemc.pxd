from posix.time cimport timeval
cimport decl

cdef extern from "../src/emc.h":
    decl.detector *det
    decl.rotation *quat
    decl.dataset *frames
    decl.iterate *iter
    decl.params *param
    
    # interp function pointers
    void (*slice_gen)(double*, double, double*, double*, long, decl.detector*) ;
    void (*slice_merge)(double*, double*, double*, double*, long, decl.detector*) ;

