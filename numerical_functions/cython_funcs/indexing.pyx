#!python
#cython: language_level=3, boundscheck=False, nonecheck=False, wraparound=False

import numpy as np
cimport numpy as np

cpdef double[:] take2(double[:] x, int[:] idx):
    cdef double[:] result = np.empty( len( idx ) )
    cdef int i
    for i in range( len( idx ) ):
        result[i] = x[ idx[ i ] ]
    return result

def take(np.ndarray[np.float64_t,ndim=1] x, np.ndarray idx):
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(idx.shape[0])
    cdef int i
    for i in range( len( idx ) ):
        result[i] = x[ idx[ i ] ]
    return result

def psquare_take( 
        np.ndarray[np.float64_t,ndim=2] source,
        np.ndarray[np.int_t,ndim=1] idx ):
    return csquare_take( source, idx )

cdef np.ndarray[np.float64_t,ndim=2] csquare_take(
        np.ndarray[np.float64_t,ndim=2] source,
        np.ndarray[np.int_t,ndim=1] idx ):
    """ Take from source, a 2d array """
    
    cdef np.ndarray[np.float64_t, ndim=2] r = np.empty( ( idx.shape[0], idx.shape[0] ) )
    cdef int i, j
    for i in range( idx.shape[0] ):
        for j in range( idx.shape[0] ):
            r[ i, j ] = source[ idx[i], idx[j] ]
    return r