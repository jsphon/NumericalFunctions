from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
# python setup.py build_ext --inplace
setup(
  name = 'Cython Functions',
  ext_modules = cythonize("*.pyx"),
  include_dirs = [np.get_include()],   
)