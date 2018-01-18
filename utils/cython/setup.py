from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_modules  =  [Extension(
    name = 'pyemc',
    sources = ['emc.pyx', '../../src/interp.c', '../../src/detector.c'],
    language = 'c',
    extra_compile_args = '-O3 -I/usr/include/python2.7 -fopenmp'.split(),
    extra_link_args = '-lpython2.7 -lm -lgomp -fopenmp'.split()
)]

setup(
    name = 'pyemc',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
