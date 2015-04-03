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


class Test(unittest.TestCase):
    
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
        
        numba_indexing.take( np.linspace( 0,1,10 ), np.array([1,2,3]))
        
        sizes = 2**np.arange(1,16)
        
        numba_timings = np.empty( sizes.shape[0] )
        cython_timings = np.empty( sizes.shape[0] )
        cython_timings2 = np.empty( sizes.shape[0] )
        numpy_timings = np.empty( sizes.shape[0] )
        
        num_tests = 100
        for i, size in enumerate( sizes ):
            x   = np.random.randn( size )
            idx = np.random.random_integers( 0, size-1, size//2 )
        
            with Timer( 'Numba' ) as numba_timer:
                for _ in range(num_tests):
                    numba_result = numba_indexing.take( x, idx )
                
            with Timer( 'Cython' ) as cython_timer:
                for _ in range(num_tests):
                    cython_result = cython_indexing.ptake( x, idx )        
            
            with Timer( 'Cython2' ) as cython_timer2:
                for _ in range(num_tests):
                    cython_result2 = cython_indexing.ptake2( x, idx )        
                
            with Timer( 'Numpy' ) as numpy_timer:
                for _ in range(num_tests):
                    numpy_result = np.take( x, idx )
                
            np.testing.assert_array_equal( numpy_result, numba_result )
            np.testing.assert_array_equal( numpy_result, cython_result )
            np.testing.assert_array_equal( numpy_result, cython_result2 )
        
            numba_timings[ i ] = numba_timer.interval
            cython_timings[ i ] = cython_timer.interval
            cython_timings2[ i ] = cython_timer2.interval
            numpy_timings[ i ] = numpy_timer.interval
            
        plt.plot( sizes, numba_timings, label='Numba' )
        plt.plot( sizes, cython_timings, label='Cython')
        plt.plot( sizes, cython_timings, label='Cython2')
        plt.plot( sizes, numpy_timings, label='Numpy' )
        plt.title( 'Take() Performance Test')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
               
    def test_square_take(self):

        X = np.random.randn( 5, 5 )
        idx = np.arange( 0, 4, 2 )
        
        cython_result = cython_indexing.psquare_take( X, idx )
        numba_result = numba_indexing.square_take( X, idx )
        numpy_result = X.take( idx, axis=0 ).take( idx, axis=1 )
        
        np.testing.assert_array_equal( numpy_result, cython_result )
        np.testing.assert_array_equal( numpy_result, numba_result )
        
    def test_square_take_performance(self):
        
        numba_indexing.square_take( np.random.randn(10,10), np.random.random_integers( 0, 5, 2 ) )
    
        sizes = 2**np.arange( 1, 8 )
        
        numba_timings = np.empty( sizes.shape[0] )
        cython_timings = np.empty( sizes.shape[0] )
        numpy_timings = np.empty( sizes.shape[0] )
        
        for i, size in enumerate( sizes ):
            x   = np.random.randn( size, size )
            idx = np.random.random_integers( 0, size-1, size//2 )
        
            with Timer( 'Numba' ) as numba_timer:
                numba_result = numba_indexing.square_take( x, idx )
                
            with Timer( 'Cython' ) as cython_timer:
                cython_result = cython_indexing.psquare_take( x, idx )        
                
            with Timer( 'Numpy' ) as numpy_timer:
                numpy_result = x.take( idx, axis=0 ).take( idx, axis=1 )
                
            np.testing.assert_array_equal( numpy_result, numba_result )
            np.testing.assert_array_equal( numpy_result, cython_result )
        
            numba_timings[ i ] = numba_timer.interval
            cython_timings[ i ] = cython_timer.interval
            numpy_timings[ i ] = numpy_timer.interval
            
        plt.plot( sizes, numba_timings, label='Numba' )
        plt.plot( sizes, cython_timings, label='Cython')
        plt.plot( sizes, numpy_timings, label='Numpy' )
        plt.title( 'Square Take() Performance Test')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()