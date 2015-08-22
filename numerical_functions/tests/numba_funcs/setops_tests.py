
import numpy as np
import unittest
import numerical_functions.numba_funcs.setops as mod

class Test(unittest.TestCase):


    def test_unique(self):
        
        x = np.array( [1, 2, 2, 3, 3, 3 ])
        r = mod.unique(x)
        
        print( r )
        
    def test_unique_inv(self):
        x = np.array( [1, 2, 2, 3, 3, 3 ])
        r = mod.unique_inv(x)
        expected = np.unique( x, return_inverse=True )

        np.testing.assert_array_equal( expected[0], r[0] )
        np.testing.assert_array_equal( expected[1], r[1] )
        
    def test_unique_inv_multi(self):
        
        for _ in range( 1000 ):
            x = np.random.random_integers( -100, 100, 100 )
            r = mod.unique_inv(x)
            expected = np.unique( x, return_inverse=True )
    
            np.testing.assert_array_equal( expected[0], r[0] )
            np.testing.assert_array_equal( expected[1], r[1] )
    
        
    def test_unique_idx(self):
        x = np.array( [1, 2, 2, 3, 3, 3 ])
        r = mod.unique_idx(x)
        print(r)
        
    def test_unique_idx2(self):
        x = np.random.random_integers(0,10,5)
        r = mod.unique_idx(x)
        
        expected = np.unique( x, return_index=True )
        print( x )
        print(r[1])
        print( expected[1] )
        
        np.testing.assert_array_equal( expected[0], r[0] )
        np.testing.assert_array_equal( expected[1], r[1] )
        
    def test_ret(self):
        
        x = np.array([1,2,3])
        r = mod.ret(x)
        
        print(r)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()