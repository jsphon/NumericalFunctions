'''
Created on 2 Apr 2015

@author: Jon
'''

import numba as nb

@nb.jit( nopython=True )
def portfolio_var( cv, weights ):
    """ Calculate portfolio variance using numba """
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