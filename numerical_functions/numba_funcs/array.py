'''
Created on 4 May 2015

@author: jon
'''


import numba as nb
import numpy as np
@nb.jit
def periods_since_not_null( x ):
    ''' x is a 1d array
    for each element of x, calculate the period since a non-null value
    0's will exist at the begining, before the first non-null value is found
    '''
    result = np.empty_like(x, dtype=np.int)
    _periods_since_not_null(x, result)
    return result
    
@nb.jit( nopython=True )
def _periods_since_not_null( x, result ):

    for i in range( x.shape[0] ):
        result[i] = 0
        if not np.isnan( x[i] ):
            break
        
    prev_real_i = i
    for i in range( prev_real_i+1, x.shape[0] ):
        result[i] = i-prev_real_i
        if not np.isnan(x[i]):
            prev_real_i=i
            
        