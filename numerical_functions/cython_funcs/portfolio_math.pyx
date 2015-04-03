#!python
#cython: language_level=3, boundscheck=False, nonecheck=False, wraparound=False

import cython
import numpy as np
cimport numpy as np

#cimport numerical_functions.cython_funcs.portfolio_math as portfolio_math

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def pportfolio_var( double[:,:] cv,
                    double[:] weights ):
    return cportfolio_var( cv, weights )
 
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def punweighted_portfolio_var( double[:,:] cv ):
    return cunweighted_portfolio_var( cv )

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def punweighted_portfolio_var_by_index( double[:,:] cv, int[:] idx ):
    return cunweighted_portfolio_var_by_index( cv, idx )

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def pportfolio_s2_by_index( cv, weights, idx ):
    return cportfolio_s2_by_index( cv, weights, idx )

