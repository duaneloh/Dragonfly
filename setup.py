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

include_dirs = [np.get_include()]
#compile_args = '-fopenmp -O3 -Wall -Wno-cpp -Wno-unused-result -Wno-unused-function -Wno-format-overflow -DFIXED_SEED'.split()
compile_args = '-fopenmp -O3 -Wall -Wno-cpp -Wno-unused-result -Wno-unused-function -Wno-format-overflow'.split() + hdf5_cflags
link_args = '-lm -fopenmp'.split() + hdf5_libs
ext_modules = [
    Extension(name='dragonfly.detector', sources=['dragonfly/detector.pyx'],
        depends=['dragonfly/detector.h', 'dragonfly/detector.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.emcfile', sources=['dragonfly/emcfile.pyx', 'dragonfly/src/emcfile.c'],
        depends=['dragonfly/emcfile.pxd', 'dragonfly/src/emcfile.h'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.model', sources=['dragonfly/model.pyx', 'dragonfly/src/model.c'],
        depends=['dragonfly/src/model.h', 'dragonfly/model.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.quaternion', sources=['dragonfly/quaternion.pyx', 'dragonfly/src/quat.c'],
        depends=['dragonfly/src/quat.h', 'dragonfly/quaternion.pxd'], include_dirs=include_dirs,
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
]
py_packages = [
    'dragonfly',
]
extensions = cythonize(ext_modules, language_level=3,
                       compiler_directives={'embedsignature': True,
                                            'boundscheck': False,
                                            'wraparound': False,
                                            'nonecheck': False})

setup(name='dragonfly',
      packages=py_packages,
      ext_modules=extensions)
