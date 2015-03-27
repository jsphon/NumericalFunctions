'''
Created on 13 Feb 2015

@author: Jon

From Numba Array Reductions notebook
'''
import numba as nb
import numpy as np


@nb.autojit( nopython=True )
def is_monotonic_increasing( x ):
    for i in range( 1, x.shape[0] ):
        if x[i]<x[i-1]:
            return False
    return True

#@nb.jit( 'f8[:](f8[:])', locals={ 'i':nb.int64 } )
@nb.autojit
def sum( x ):
    r=0
    for i in range( x.shape[0] ):
        r+=x[i]
    return r

@nb.autojit
def sum_cols( X ):
    r = np.zeros( X.shape[1], dtype=X.dtype )
    for i in range( X.shape[0] ):
        for j in range( X.shape[1] ):
            r[j]+=X[i,j]
    return r

@nb.autojit
def mean_cols( X ):
    return sum_cols( X ) / X.shape[0]

@nb.autojit
def std_cols( X ):
    m = mean_cols( X )
    d = np.zeros( X.shape[1], dtype=X.dtype )
    r = np.zeros( X.shape[1], dtype=X.dtype )
    
    for i in range( X.shape[0] ):
        for j in range( X.shape[1] ):
            d[j] += (X[i,j]-m[j] ) **2
    
    for j in range( X.shape[1] ):
        r[j] = np.sqrt( d[j] / X.shape[0] )
        
    return r

@nb.autojit
def standardised_mean_cols( X ):
    m = mean_cols( X )
    d = np.zeros( X.shape[1], dtype=X.dtype )
    
    for i in range( X.shape[0] ):
        for j in range( X.shape[1] ):
            d[j] += (X[i,j]-m[j] ) **2
    
    for j in range( X.shape[1] ):
        d[j] = np.sqrt( d[j] / X.shape[0] )
        if d[j]!=0:
            m[j]=m[j]/d[j]
        else:
            m[j] = np.nan
    return m
  
