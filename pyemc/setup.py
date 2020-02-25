from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import os
import sys
import subprocess
import glob
import numpy as np
try:
    import h5py
    out = subprocess.getoutput('h5cc -shlib -show')
    hdf5_cflags = ['-DWITH_HDF5', [s for s in out.split() if s[:2] == '-I'][0]]
    hdf5_libs = [[s for s in out.split() if s[:2] == '-L'][0], '-lhdf5']
except ImportError:
    hdf5_cflags = []
    hdf5_libs = []

gsl_cflags = subprocess.check_output('gsl-config --cflags', shell=True).decode(sys.stdout.encoding).rstrip().split()
gsl_libs = subprocess.check_output('gsl-config --libs', shell=True).decode(sys.stdout.encoding).rstrip().split()
compile_args = '-fopenmp -O3 -Wall -Wno-cpp -Wno-unused-result -Wno-unused-function -Wno-format-overflow -DFIXED_SEED'.split() + gsl_cflags + hdf5_cflags
include_dirs = [np.get_include()]
link_args = '-lm -fopenmp'.split() + gsl_libs + hdf5_libs
#mpi_compile_args = subprocess.check_output('mpicc --showme:compile', shell=True).decode(sys.stdout.encoding).rstrip().split()
#mpi_link_args = subprocess.check_output('mpicc --showme:link', shell=True).decode(sys.stdout.encoding).rstrip().split()
mpi_compile_args = subprocess.check_output('mpicc -show', shell=True).decode(sys.stdout.encoding).rstrip().split()[1:]
mpi_link_args = subprocess.check_output('mpicc -show', shell=True).decode(sys.stdout.encoding).rstrip().split()[1:]

ext_modules = [
    Extension(name='detector', sources='detector.pyx ../src/detector.c'.split(),
        depends='../src/detector.h decl.pxd'.split(), include_dirs=include_dirs,
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='quat', sources='quat.pyx ../src/quat.c'.split(),
        depends='../src/quat.h decl.pxd'.split(), include_dirs=include_dirs,
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dataset', sources='dataset.pyx ../src/dataset.c'.split(),
        depends='../src/dataset.h decl.pxd detector.pxd'.split(), include_dirs=include_dirs,
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='interp', sources='interp.pyx ../src/interp.c'.split(),
        depends='../src/interp.h decl.pxd detector.pxd'.split(), include_dirs=include_dirs,
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='params', sources='params.pyx ../src/params.c'.split(),
        depends='../src/params.h decl.pxd'.split(), include_dirs=include_dirs,
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='iterate', sources='iterate.pyx ../src/iterate.c ../src/quat.c ../src/dataset.c'.split(),
        depends='../src/iterate.h decl.pxd detector.pxd dataset.pxd params.pxd quat.pxd'.split(), include_dirs=include_dirs,
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='max_emc', sources='max_emc.pyx ../src/setup_emc.c ../src/output_emc.c'.split(),
        extra_objects=['build/src/'+f for f in 'detector.o quat.o dataset.o interp.o iterate.o params.o'.split()],
        depends='../src/emc.h ../src/max_emc.c'.split(), include_dirs=include_dirs,
        language='c', extra_compile_args=compile_args+mpi_compile_args, extra_link_args=link_args+mpi_link_args),
    Extension(name='pyemc', sources=['pyemc.pyx', '../src/setup_emc.c', '../src/output_emc.c'],
        extra_objects=['build/src/'+f for f in 'detector.o quat.o dataset.o interp.o iterate.o params.o'.split()],
        depends=glob.glob('../src/*emc*') + glob.glob('*.pxd'), include_dirs=include_dirs,
        language='c', extra_compile_args=compile_args+mpi_compile_args, extra_link_args=link_args+mpi_link_args),
]

setup(name='pyemc', cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
