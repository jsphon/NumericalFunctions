'''
Created on 27 Mar 2015

@author: Jon
'''

from numba_funcs.timer import Timer
import numba_funcs.indexing as indexing
import numpy as np
import unittest


class Test(unittest.TestCase):
        
    def test_take(self):
        x = np.linspace( 0, 100 )
        idx = np.random.random_integers( 0, 50, 20 )
        result = indexing.take( x, idx )
        expected = np.take( x, idx )
        np.testing.assert_array_equal( expected, result )
        
    def test_take_comparison(self):
        x = np.arange( 1e6 )
        idx = np.random.random_integers( 0, 1e5, 1e6 )
        result = indexing.take( x, idx )
        expected = np.take( x, idx )
        
        with Timer( 'numba' ) as nbtimer:
            result = indexing.take( x, idx )
            
        with Timer( 'numpy' ) as nptimer:
            expected = np.take( x, idx )
           
        ratio = nbtimer.interval / nptimer.interval
        print( 'numba version of take took %0.2f as long as numpy'%ratio)        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()