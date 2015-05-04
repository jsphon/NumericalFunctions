'''
Created on 4 May 2015

@author: jon
'''
import unittest

import numerical_functions.numba_funcs.array as array
import numpy as np

class Test(unittest.TestCase):


    def test_periods_since_not_null(self):
        x = np.array( [ 0, 1, 2, 3 ] )
        result = array.periods_since_not_null(x)
        
        expected = np.array( [ 0, 1, 1, 1 ] )
        
        np.testing.assert_array_equal( expected, result )
        
    def test_periods_since_not_null2(self):
        x = np.array( [ 0, 1, np.nan, 3 ] )
        result = array.periods_since_not_null(x)
        
        expected = np.array( [ 0, 1, 1, 2 ] )
        
        np.testing.assert_array_equal( expected, result )
        
    def test_periods_since_not_null3(self):
        x        = np.array( [ np.nan, 1, np.nan, 3 ] )
        expected = np.array( [ 0     , 0, 1     , 2 ] )
        result = array.periods_since_not_null(x)
        np.testing.assert_array_equal( expected, result )
        
    def test_periods_since_not_null4(self):
        x        = np.array( [ np.nan, np.nan, np.nan, 3 ] )
        expected = np.array( [ 0     , 0     , 0     , 0 ] )
        result = array.periods_since_not_null(x)
        np.testing.assert_array_equal( expected, result )
        
    def test_periods_since_not_null5(self):
        x        = np.array( [ 0, np.nan, np.nan, 3, 4 ] )
        expected = np.array( [ 0, 1     , 2     , 3, 1 ] )
        result = array.periods_since_not_null(x)
        np.testing.assert_array_equal( expected, result )



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()