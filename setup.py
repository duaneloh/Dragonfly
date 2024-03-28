from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import subprocess
import numpy as np
import h5py
out = subprocess.getoutput('h5cc -shlib -show')
hdf5_cflags = [[s for s in out.split() if s[:2] == '-I'][0]]
hdf5_libs = [[s for s in out.split() if s[:2] == '-L'][0], '-lhdf5']

mpi_cflags = subprocess.getoutput('mpicc --showme:compile').strip().split()
mpi_libs = subprocess.getoutput('mpicc --showme:link').strip().split()
gsl_cflags = subprocess.getoutput('gsl-config --cflags').strip().split()
gsl_libs = subprocess.getoutput('gsl-config --libs').strip().split()

compile_args = '-fopenmp -O3 -Wall'.split()
compile_args += '-Wno-cpp -Wno-unused-result -Wno-unused-function -Wno-format-overflow'.split() 
compile_args += hdf5_cflags + gsl_cflags + ['-I'+np.get_include()]
link_args = '-lm -fopenmp'.split() + hdf5_libs + gsl_libs + ['-Wl,-rpath='+gsl_libs[0][2:]]

ext_modules = [
    Extension(name='dragonfly.detector', sources=['dragonfly/detector.pyx'],
        depends=['dragonfly/src/detector.h', 'dragonfly/detector.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.emcfile', sources=['dragonfly/emcfile.pyx', 'dragonfly/src/emcfile.c'],
        depends=['dragonfly/src/emcfile.h', 'dragonfly/emcfile.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.model', sources=['dragonfly/model.pyx', 'dragonfly/src/model.c'],
        depends=['dragonfly/src/model.h', 'dragonfly/model.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.quaternion', sources=['dragonfly/quaternion.pyx', 'dragonfly/src/quaternion.c'],
        depends=['dragonfly/src/quaternion.h', 'dragonfly/quaternion.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.iterate', sources=['dragonfly/iterate.pyx', 'dragonfly/src/iterate.c'],
        depends=['dragonfly/src/iterate.h', 'dragonfly/iterate.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.params', sources=['dragonfly/params.pyx'],
        depends=['dragonfly/src/params.h', 'dragonfly/params.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.recon', sources=['dragonfly/recon.pyx', 'dragonfly/src/maximize.c', 'dragonfly/src/model.c'],
        depends=['dragonfly/src/maximize.h', 'dragonfly/recon.pxd'],
        language='c', extra_compile_args=compile_args+mpi_cflags, extra_link_args=link_args+mpi_libs),
]
py_packages = [
    'dragonfly',
]
extensions = cythonize(ext_modules, language_level=3,
                       compiler_directives={'embedsignature': True,
                                            'boundscheck': False,
                                            'wraparound': False,
                                            'cdivision': True,
                                            'nonecheck': False})

with open('dragonfly/_version.py', 'r') as f:
    exec(f.read())

setup(name='dragonfly',
      version=__version__,
      packages=py_packages,
      ext_modules=extensions,
      entry_points = {'console_scripts': [
          'pyemc = dragonfly.recon:main',
          ]
      },
)
