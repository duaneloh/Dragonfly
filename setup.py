from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

include_dirs = [np.get_include()]
#compile_args = '-fopenmp -O3 -Wall -Wno-cpp -Wno-unused-result -Wno-unused-function -Wno-format-overflow -DFIXED_SEED'.split()
compile_args = '-fopenmp -O3 -Wall -Wno-cpp -Wno-unused-result -Wno-unused-function -Wno-format-overflow'.split()
link_args = '-lm -fopenmp'.split()
ext_modules = [
    Extension(name='dragonfly.detector', sources=['dragonfly/detector.pyx'],
        depends=['dragonfly/detector.h', 'dragonfly/detector.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.emcfile', sources=['dragonfly/emcfile.pyx'],
        depends=['dragonfly/emcfile.pxd'],
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
    Extension(name='dragonfly.model', sources=['dragonfly/model.pyx', 'dragonfly/src/model.c'],
        depends=['dragonfly/model.h', 'dragonfly/model.pxd'],
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
