'''
Created on 27 Mar 2015

@author: Jon
'''

import numerical_functions.numba_funcs as nf
from numerical_functions.misc.timer import Timer
import numpy as np
import unittest


class Test(unittest.TestCase):

    def test_binary_search(self):
        for _ in range( 100 ):
            x = np.random.random_integers( 0, 100, 10 )
            x.sort()    
            
            r = nf.binary_search( x, x[5] )
            expected = np.searchsorted( x, x[5] )
            
            if expected!=r:
                print( 'x[5]=%s'%x[5])
                print( 'x=%s'%str(x) )
                print( 'r=%s'%r )
                print( 'expected=%s'%str(expected ))
            self.assertEqual( expected, r )
            
    def test_binary_search_not_found(self):

        x = np.random.random_integers( 0, 100, 10 )
        x.sort()    
        
        v = -1
        
        r = nf.binary_search( x, v )
        expected = np.searchsorted( x, v )
        
        print( 'r=%s'%r )
        self.assertEqual( expected, r )
            
    def test_binary_search_performance(self):
        
        x = np.random.random_integers( 0, 100000, 10000 )
        x.sort()    
            
        nf.binary_search( x, x[500] )
        
        num_tests = 1000
        
        with Timer( 'nb' ):
            for _ in range( num_tests ):            
                nf.binary_search( x, x[5000] )
                
        with Timer( 'np' ):
            for _ in range( num_tests ):
                np.searchsorted( x, x[5000] )
                
    def test_quick_sort(self):
        x = np.random.random_integers( 0, 100000, 10000 )
        
        self.assertFalse( nf.is_monotonic_increasing( x ) )
        nf.quick_sort( x )        
        self.assertTrue( nf.is_monotonic_increasing( x ) )
                
    def test_quick_sort_performance(self):
        x = np.random.random_integers( 0, 100000, 100000 )
        
        x0 = x.copy()
        x1 = x.copy()
        
        nf.quick_sort( x0.copy() ) 
        with Timer( 'nb' ):
            nf.quick_sort( x0 )

        with Timer( 'np' ):
            np.sort( x1 )
            
    def test_quick_arg_sort(self):
        
        x     = np.array( [ 3,2,1,4 ] )
        x0    = x.copy()
        args0 = nf.quick_arg_sort(x0)
        
        x1    = x.copy() 
        args1 = np.argsort( x1 )
        
        np.testing.assert_equal( args0, args1 )
        
            
    def test_quick_arg_partition1(self):
        
        x   = np.array( [ 3, 1, 2, 4 ] )
        arg = np.arange( x.shape[0] )
        
        nf.quick_arg_partition(x, arg, 0, x.shape[0]-1 )
        
        np.testing.assert_array_equal( [ 1, 2, 3, 4], x )
        np.testing.assert_array_equal( [ 2, 1, 0, 3 ], arg )
        
    def test_quick_arg_partition2(self):
        
        x   = np.array( [ 2, 3, 1, 4 ] )
        arg = np.arange( x.shape[0] )
        
        nf.quick_arg_partition(x, arg, 0, x.shape[0]-1 )
        
        np.testing.assert_array_equal( [ 1, 2, 3, 4], x )
        np.testing.assert_array_equal( [ 2, 0, 1, 3 ], arg )