#!python
#cython: language_level=3, boundscheck=False, nonecheck=False, wraparound=False

import cython
import numpy as np
cimport numpy as np

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def add( double a, double b ):
    return a+b