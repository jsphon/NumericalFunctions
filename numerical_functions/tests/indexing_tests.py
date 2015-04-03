import numpy as np
import unittest

import pyximport
pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
                              "include_dirs":np.get_include()},
                  reload_support=True)

import matplotlib.pyplot as plt
from numerical_functions.misc.timer import Timer
import numerical_functions.cython_funcs.indexing as cython_indexing
import numerical_functions.numba_funcs.indexing as numba_indexing
import numerical_functions.numpy_funcs.indexing as numpy_indexing

from numerical_functions.tests.function_comparer import FunctionComparer

class Test( FunctionComparer ):
    
    def setUp(self):
        self.sizes = 2**np.arange(1,10)        
        
    def test_take(self):
        x = np.linspace( 0, 100 )
        idx = np.random.random_integers( 0, 50, 20 )
        
        cython_result = cython_indexing.ptake( x, idx )
        numba_result = numba_indexing.take( x, idx )
        expected = np.take( x, idx )
        
        np.testing.assert_array_equal( expected, cython_result )
        np.testing.assert_array_equal( expected, numba_result )
        
    def test_take_performance(self):
        sizes = 2**np.arange( 1, 20 )
        
        fns = {
            'Cython' : cython_indexing.ptake,
            'Numba' : numba_indexing.take,
            'Numpy' : np.take            
       }
        
        self.compare_performance( 'Take', fns, _gen_take_args, sizes )
        
    def test_take_to_out(self):
        
        x = np.linspace( 0, 100,100 )
        idx = np.random.random_integers( 0, 50, 20 )
        
        cython_result = np.empty_like( idx, dtype=x.dtype )
        numba_result  = np.empty_like( idx, dtype=x.dtype ) 
        
        cython_indexing.ptake_to_out( x, idx, cython_result )
        numba_indexing.take_to_out( x, idx, numba_result )        
        expected = np.take( x, idx )
        
        np.testing.assert_array_equal( expected, cython_result )
        np.testing.assert_array_equal( expected, numba_result )
        
    def test_take_to_out_performance(self):
        sizes = 2**np.arange( 1, 20 )
        
        fns = {
            'Cython' : cython_indexing.ptake_to_out,
            'Numba' : numba_indexing.take_to_out,           
       }
        
        self.compare_performance( 'Square Take To Out', fns, _gen_take_to_out_args, sizes )
               
    def test_square_take(self):

        X = np.random.randn( 5, 5 )
        idx = np.arange( 0, 4, 2 )
        
        cython_result = cython_indexing.psquare_take( X, idx )
        numba_result = numba_indexing.square_take( X, idx )
        numpy_result = numpy_indexing.square_take( X, idx )
        
        np.testing.assert_array_equal( numpy_result, cython_result )
        np.testing.assert_array_equal( numpy_result, numba_result )        
    
    def test_square_take_performance(self):
        sizes = 2**np.arange( 1, 10 )
        
        fns = {
            'Cython' : cython_indexing.psquare_take,
            'Numba' : numba_indexing.square_take,
            'Numpy' : numpy_indexing.square_take            
       }
        
        self.compare_performance( 'Square Take', fns, _gen_square_take_args, sizes )

def _gen_take_to_out_args( msize ):
    x   = np.random.randn( msize )
    idx = np.random.random_integers( 0, msize-1, msize//2 )
    out = np.empty_like( idx, dtype=x.dtype )
    return x, idx, out
      
def _gen_take_args( msize ):
    x   = np.random.randn( msize )
    idx = np.random.random_integers( 0, msize-1, msize//2 )
    return x, idx
            
def _gen_square_take_args( msize ):
    x   = np.random.randn( msize, msize )
    idx = np.random.random_integers( 0, msize-1, msize//2 )
    return x, idx
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()