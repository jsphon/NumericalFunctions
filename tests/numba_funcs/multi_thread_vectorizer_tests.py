# -*- coding: utf-8 -*-

from numba import vectorize

import numpy as np
from numba import jit, void, double

from numba_funcs.multi_thread_vectorizer import mvectorize
from numba_funcs.timer import Timer

import unittest

"""
def fn( x ):
    result = 0.0
    for i in range( 100 ):
        result += (1+i+x)/(1+x)
    return result
""" 

def fn( x ):
    result = 0.0
    for i in range( 10000 ):
        result+=i*x
    return result

nb_fn = jit( double(double, ), nopython=True )( fn )

class Test(unittest.TestCase):

    def test_mvectorize(self):
        
        x = np.linspace( 1, 1000, 10000 )
        
        mf_fn = mvectorize( nb_fn, ( double[:], double[:] ), num_threads=8 )
        
        result = mf_fn( x )
        expected = np.vectorize( fn )( x )
        
        np.testing.assert_array_equal( expected, result )
        
    def test_mvectorize_performance(self):
        
        x = np.linspace( 0, 1000 )
        
        mf_fn = mvectorize( nb_fn, ( double[:], double[:] ), num_threads=8 )
        # Call once to initialize
        mf_fn( x )
        num_tests = 100 
        
        with Timer( 'mf_nb' ) as nbtimer:
            for _ in range( num_tests ):
                mf_fn( x )
                
        np_fn = np.vectorize( fn )
        with Timer( 'np' ) as nptimer:
            for _ in range( num_tests ):
                np_fn( x )
                
        ratio = nbtimer.interval / nptimer.interval
        print( 'Numba version took %s as long as numpy'%ratio)
        
            
            