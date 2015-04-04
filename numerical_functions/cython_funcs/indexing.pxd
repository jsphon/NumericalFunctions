#!python
#cython: language_level=3, boundscheck=False, nonecheck=False, wraparound=False

import cython
cimport cpython.array

import numpy as np
cimport numpy as np

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cdef inline void cswap_row_cols( double[:, :] X, int i, int j ):
    """ Swap the rows and cols of X indexed by i and j """
    cdef:
        int a
        double t

    for a in range( X.shape[0] ):
        t = X[ a, j ]
        X[ a, j ] = X[ a, i ]
        X[ a, i ] = t
        
    for a in range( X.shape[0] ):
        t = X[ j, a ]
        X[ j, a ] = X[ i, a ]
        X[ i, a ] = t
            