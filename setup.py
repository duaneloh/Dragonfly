from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

include_dirs = [np.get_include()]
compile_args = '-fopenmp -O3 -Wall -Wno-cpp -Wno-unused-result -Wno-unused-function -Wno-format-overflow -DFIXED_SEED'.split()
link_args = '-lm -fopenmp'.split()
ext_modules = [
    Extension(name='dragonfly.detector', sources=['dragonfly/detector.pyx'],
              depends=['dragonfly/detector.pxd']),
    Extension(name='dragonfly.emcfile', sources=['dragonfly/emcfile.pyx']),
    Extension(name='dragonfly.model', sources=['dragonfly/model.pyx']),
    Extension(name='dragonfly.quaternion', sources='dragonfly/quaternion.pyx dragonfly/src/quat.c'.split(),
        depends='dragonfly/src/quat.h dragonfly/quaternion.pxd'.split(), include_dirs=include_dirs,
        language='c', extra_compile_args=compile_args, extra_link_args=link_args),
]
py_packages = [
    'dragonfly',
]
extensions = cythonize(ext_modules, language_level=3,
                       compiler_directives={'embedsignature': True})

setup(name='dragonfly',
      packages=py_packages,
      ext_modules=extensions)
