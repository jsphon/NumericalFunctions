#!python
#cython: language_level=3, boundscheck=False, nonecheck=False, wraparound=False

import cython
import numpy as np
cimport numpy as np

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def pportfolio_var( double[:,:] cv,
                    double[:] weights ):
    return cportfolio_var( cv, weights )
 
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cdef double cportfolio_var( double[:,:] cv,
                  double[:] weights):
    """ Calculate portfolio variance"""
    cdef double s0=0.0
    cdef double s1=0.0
    cdef double s2
    cdef size_t i, j
    #s0 = 0.0
    for i in range( weights.shape[0] ):
        s0 += weights[i]*weights[i]*cv[i,i]
        
    #s1 = 0.0
    for i in range( weights.shape[0]-1 ):
        s2 = 0.0
        for j in range( i+1, weights.shape[0] ):
            s2 += weights[j]*cv[i,j]
        s1+= weights[i]*s2
    return s0+2.0*s1

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def punweighted_portfolio_var( double[:,:] cv ):
    return cunweighted_portfolio_var( cv )
                                             
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def cunweighted_portfolio_var( double[:,:] cv ):
    """ Calculate unweighted
    Divide by cv.shape[0]**2 to get the even weighted portfolio variance
    """
    cdef double s0=0.0
    cdef double s1=0.0
    cdef size_t i
    for i in range( cv.shape[0] ):
        s0 += cv[i,i]
        
    for i in range( cv.shape[0]-1 ):        
        for j in range( i+1, cv.shape[0] ):
            s1 += cv[i,j]        
    return s0+2.0*s1  


def pportfolio_s2_by_index( cv, weights, idx ):
    return cportfolio_s2_by_index( cv, weights, idx )

cdef np.float64_t cportfolio_s2_by_index( 
         np.ndarray[np.float64_t,ndim=2] cv,
         np.ndarray[np.float64_t,ndim=1] weights,
         np.ndarray[np.int64_t,ndim=1] idx ):
    """ Calculate portfolio variance using numba """
    cdef double s0=0.0, s1=0.0, s2
    cdef int i, ii, j, jj
    for i in range( idx.shape[0] ):
        j = idx[i]
        s0 += weights[j]*weights[j]*cv[j,j]
        
    for i in range( idx.shape[0]-1 ):
        ii = idx[i]
        s2 = 0.0        
        for j in range( i+1, idx.shape[0] ):
            jj = idx[j]
            s2 += weights[jj]*cv[ii,jj]
        s1+= weights[ii]*s2
    return s0+2.0*s1 