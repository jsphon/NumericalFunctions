'''
Created on 27 Mar 2015

@author: Jon
'''

import numba_funcs.array_reductions as ar
import numpy as np
import unittest


class Test(unittest.TestCase):


    def test_is_monotonic_increasing_true(self):
        x = np.linspace( 0, 10 )
        self.assertTrue( ar.is_monotonic_increasing( x ) )
        
    def test_is_monotonic_increasing_true2(self):
        x = np.array( [0, 0, 1 ] )
        self.assertTrue( ar.is_monotonic_increasing( x ) )
        
    def test_is_monotonic_increasing_false(self):
        x = np.array( [0, 0, -1 ] )
        self.assertFalse( ar.is_monotonic_increasing( x ) )
        
    def test_sum(self):
        x = np.arange(10)
        result = ar.sum( x )
        expected = 10*9/2
        self.assertEqual( expected, result )
        
    def test_sum_cols(self):
        x = np.arange(10).reshape(5,2)
        result = ar.sum_cols( x )
        expected = x.sum( axis=0 )
        np.testing.assert_array_equal( expected, result )
        
    def test_mean_cols(self):
        x = np.random.random((10,3))
        result = ar.mean_cols(x)
        expected = x.mean( axis=0 )
        np.testing.assert_array_equal( expected, result )
        
    def test_std_cols(self):
        x = np.random.random((10,3))
        result = ar.std_cols(x)
        expected = np.std( x, axis=0 )
        np.testing.assert_array_equal( expected, result )
        
    def test_standardised_mean_cols(self):
        x = np.random.random((10,3))
        result = ar.standardised_mean_cols(x)
        expected = np.mean( x, axis=0 ) / np.std( x, axis=0 )
        np.testing.assert_array_equal( expected, result )
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()