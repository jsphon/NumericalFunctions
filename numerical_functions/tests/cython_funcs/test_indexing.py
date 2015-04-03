'''
Created on 29 Mar 2015

@author: Jon
'''

import matplotlib.pyplot as plt
import numpy as np
import unittest
import pyximport
#pyximport.install()

pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
                              "include_dirs":np.get_include()},
                  reload_support=True)

import numerical_functions.cython_funcs.indexing as indexing
import numerical_functions.numba_funcs.indexing as indexing_nb

from numerical_functions.misc.timer import Timer

class Test(unittest.TestCase):

    def test_take(self):
        
        x = np.linspace( 0, 1, 100 )
        idx = np.array( [ 3,2,1 ] )
        
        cresult = indexing.take( x, idx )
        nresult = indexing_nb.take( x, idx )
        
        print( cresult )
        print( nresult )
        
        np.testing.assert_array_equal( nresult, cresult )
               
    def test_take_performance(self):
        
        N = 10
        x  = np.linspace( 0, 100, N )
        idx = np.random.random_integers( 0, N-1, N//2 )
        
        indexing.take( x, idx )
        indexing_nb.take( x, idx )
        
        num_tests = 1000
        
        with Timer( 'Cython' ) as cython_timer:
            for _ in range( num_tests ):
                indexing.take( x, idx )
            
        with Timer( 'Numba' ) as numba_timer:
            for _ in range( num_tests ):
                indexing.take( x, idx )
        
        ratio = cython_timer.interval / numba_timer.interval
        print( 'Cython took %s as long as Numba.'%ratio )
        
    def test_psquare_take(self):
        
        x = np.random.randn(10,10)
        idx = np.array( [ 3,2,1 ] )
        
        cresult = indexing.psquare_take( x, idx )
        nresult = indexing_nb.square_take( x, idx )
        
        np.testing.assert_array_equal( nresult, cresult )
        
    def test_psquare_take_performance(self):
        
        sizes = 2**np.arange( 1, 10 )
        cython_times = []
        numba_times = []
        
        numba_timer = Timer()
        cython_timer = Timer()
        
        indexing_nb.square_take( np.random.randn( 2,2 ), np.array([0,1]) )
        
        for i, size in enumerate( sizes ):
            print( 'Analysing size %s / %s'%(i,size))
            x = np.random.randn( size, size )
            idx = np.random.random_integers( 0, size-1, size//2 )
            
            with cython_timer:
                cresult = indexing.psquare_take( x, idx )
            cython_times.append( cython_timer.interval )
            
            with numba_timer:
                nresult = indexing_nb.square_take( x, idx )
            numba_times.append( numba_timer.interval )
            
            np.testing.assert_array_equal( nresult, cresult )
            
        plt.plot( sizes, cython_times, label='Cython' )        
        plt.plot( sizes, numba_times, label='Numba' )
        plt.title( 'Execution Time By Size Size' )
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()