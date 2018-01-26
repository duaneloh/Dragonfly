from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import subprocess
import glob

gsl_cflags = subprocess.check_output('gsl-config --cflags', shell=True).rstrip().split()
gsl_libs = subprocess.check_output('gsl-config --libs', shell=True).rstrip().split()
compile_args = '-I/usr/include/python2.7 -fopenmp -O3 -Wall -Wno-cpp -Wno-unused-result -Wno-unused-function'.split() + gsl_cflags
link_args = '-lpython2.7 -lm -fopenmp'.split() + gsl_libs
mpi_compile_args = subprocess.check_output('mpicc --showme:compile', shell=True).rstrip().split()
mpi_link_args = subprocess.check_output('mpicc --showme:link', shell=True).rstrip().split()

ext_modules = [
    Extension(name='detector', sources='detector.pyx ../src/detector.c'.split(),
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='quat', sources='quat.pyx ../src/quat.c'.split(),
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dataset', sources='dataset.pyx ../src/dataset.c'.split(),
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='interp', sources='interp.pyx ../src/interp.c'.split(),
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='params', sources='params.pyx ../src/params.c'.split(),
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='iterate', sources='iterate.pyx ../src/iterate.c'.split(),
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='pyemc', sources=['pyemc.pyx']+glob.glob('../src/*emc.c'), 
        extra_objects=['build/src/'+f for f in 'detector.o quat.o dataset.o interp.o iterate.o params.o'.split()],
        language='c', extra_compile_args=compile_args+mpi_compile_args, extra_link_args=link_args+mpi_link_args),
]

setup(name='pyemc', cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
