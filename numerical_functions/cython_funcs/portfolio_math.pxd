import cython

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cdef inline double cportfolio_var( double[:,:] cv,
                  double[:] weights):
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
cdef inline double cunweighted_portfolio_var( double[:,:] cv ):
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

@cython.wraparound(True)
@cython.nonecheck(True)
@cython.boundscheck(True)
cdef inline double cunweighted_portfolio_var_by_index( double[:,:] cv, int[:] idx ):
    """ Calculate unweighted
    Divide by cv.shape[0]**2 to get the even weighted portfolio variance
    Only use elements of cv specified by idx
    """
    cdef double s0=0.0
    cdef double s1=0.0
    cdef size_t i, idxi
    for i in range( idx.shape[0] ):
        idxi = idx[i]
        s0 += cv[idxi,idxi]
        
    s1 = 0.0 
    for i in range( idx.shape[0]-1 ):
        idxi = idx[i]        
        for j in range( i+1, idx.shape[0] ):            
            s1 += cv[ idxi, idx[j] ]    
      
    return s0+2.0*s1  

@cython.wraparound(True)
@cython.nonecheck(True)
@cython.boundscheck(True)
cdef inline double cportfolio_s2_by_index( 
         double[:,:] cv,
         double[:] weights,
         int[:] idx ):
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