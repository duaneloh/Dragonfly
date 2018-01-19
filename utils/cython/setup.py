from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import subprocess

gsl_cflags = subprocess.check_output('gsl-config --cflags', shell=True).rstrip().split()
gsl_libs = subprocess.check_output('gsl-config --libs', shell=True).rstrip().split()

ext_modules  =  [Extension(
    name = 'pyemc',
    sources = ['emc.pyx', '../../src/interp.c', '../../src/detector.c', '../../src/dataset.c'],
    language = 'c',
    extra_compile_args = '-I/usr/include/python2.7 -fopenmp -O3 -Wall -Wno-cpp -Wno-unused-result -Wno-unused-function'.split() + gsl_cflags,
    extra_link_args = '-lpython2.7 -lm -fopenmp'.split() + gsl_libs
)]

setup(
    name = 'pyemc',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
