cdef extern from 'src/quaternion.h' nogil:
    struct quaternion:
        int num_div, num_rot, num_rot_p
        double *quats
        int reduced, icosahedral_flag, octahedral_flag

    int quat_gen(int, quaternion*)
    void voronoi_subset(quaternion*, quaternion*, int*)

cdef class Quaternion:
    cdef quaternion* quat
