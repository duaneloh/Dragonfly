cdef extern from 'src/quat.h' nogil:
    struct quaternion:
        int num_div, num_rot, num_rot_p
        double *quats
        int icosahedral_flag, cubic_flag

    int quat_gen(int, quaternion*)

cdef class Quaternion:
    cdef quaternion* quat
