from libc.stdint cimport uint8_t
from .detector cimport detector

cdef extern from "src/model.h" nogil:
    cdef enum model_type: MODEL_3D, MODEL_2D, MODEL_RZ

    struct model:
        model_type mtype
        long size, center, vol
        int num_modes
        double *model1
        double *model2
        double *inter_weight

    void make_rot_quat(double*, double[3][3])
    void make_rot_angle(double, double[2][2])
    void slice_gen3d(double*, int, double*, detector*, model*)
    void slice_gen2d(double*, int, double*, detector*, model*)
    void slice_genrz(double*, int, double*, detector*, model*)
    void slice_merge3d(double*, int, double*, detector*, model*)
    void slice_merge2d(double*, int, double*, detector*, model*)
    void slice_mergerz(double*, int, double*, detector*, model*)
    void rotate_model(double[3][3], double*, int, int, double*)
    void symmetrize_icosahedral(double*, int)
    void symmetrize_octahedral(double*, int)
    void symmetrize_friedel(double*, int)
    void symmetrize_friedel2d(double*, int, int)

cdef class Model:
    cdef model* mod
