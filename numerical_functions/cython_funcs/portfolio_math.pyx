#!python
#cython: language_level=3, boundscheck=False, nonecheck=False, wraparound=False

import numpy as np
cimport numpy as np

 
cpdef double portfolio_s2_call( np.ndarray[np.float64_t,ndim=2] cv,
                  np.ndarray[np.float64_t,ndim=1] weights):
    return 0.0
 
cpdef double portfolio_s2( np.ndarray[np.float64_t,ndim=2] cv,
                  np.ndarray[np.float64_t,ndim=1] weights):
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

def portfolio_s2_opt(double[:,:] cv, double[:] weights):    
    """ Calculate portfolio variance using memory views"""
    cdef double s0
    cdef double s1
    cdef double s2
    cdef size_t i, j

    s0 = 0.0
    for i in range( weights.shape[0] ):
        s0 += weights[i]*weights[i]*cv[i,i]

    s1 = 0.0
    for i in range( weights.shape[0]-1 ):
        s2 = 0.0
        for j in range( i+1, weights.shape[0] ):
            s2 += weights[j]*cv[i,j]
        s1+= weights[i]*s2
    return s0+2.0*s1

def pportfolio_s2_by_index( cv, weights, idx ):
    return portfolio_s2_by_index( cv, weights, idx )

cdef np.float64_t portfolio_s2_by_index( 
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