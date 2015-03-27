'''
Created on 13 Feb 2015

@author: Jon

From Numba Array Reductions notebook
'''
import numba as nb
import numpy as np

@nb.autojit
def take( x, idx ):
    r = np.ndarray( idx.shape[0], dtype=x.dtype )
    for i in range( idx.shape[0] ):
        r[i]=x[ idx[ i ] ]
    return r