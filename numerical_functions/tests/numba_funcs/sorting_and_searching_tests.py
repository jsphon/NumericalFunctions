'''
Created on 27 Mar 2015

@author: Jon
'''

import numerical_functions.numba_funcs as nf
import numerical_functions.numba_funcs.sorting_and_searching as sas
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
        
    def test__merge1(self):
        
        x = np.array( [ 1, 3, 2, 4 ] )
        
        y = np.empty_like( x )
        sas._merge( x, y, 0, 2, 4 )
        
        e = np.sort( x )
        np.testing.assert_array_equal( e, y ) 
        
    def test__merge2(self):
        
        x = np.array( [ 1, 2 ] )
        
        y = np.empty_like( x )
        sas._merge( x, y, 0, 1, 2 )
        
        e = np.sort( x )
        np.testing.assert_array_equal( e, y ) 
        
    def test__merge3(self):
        
        x = np.array( [ 2, 1 ] )
        
        y = np.empty_like( x )
        sas._merge( x, y, 0, 1, 2 )
        
        e = np.sort( x )
        np.testing.assert_array_equal( e, y ) 
        
    def test__merge4(self):
        
        x = np.array( [ 2, 1, 3 ] )
        
        y = np.empty_like( x )
        sas._merge( x, y, 0, 1, 3 )
        
        e = np.sort( x )
        np.testing.assert_array_equal( e, y )
        
    def test__merge5(self):
        
        x = np.array( [ 1, 2, 0 ] )
        
        y = np.empty_like( x )
        sas._merge( x, y, 0, 2, 3 )
        
        e = np.sort( x )
        np.testing.assert_array_equal( e, y )
        
    def test__merge6(self):
        
        x = np.array( [ 1, 1, 0 ] )
        
        y = np.empty_like( x )
        sas._merge( x, y, 0, 2, 3 )
        
        e = np.sort( x )
        np.testing.assert_array_equal( e, y )
        
        
    def test_mergeb(self):
        
        x = np.array( [ 3,4,1,2 ] )
        
        
        y = sas.merge2( x )
        
        e = np.sort( x )
        np.testing.assert_array_equal( e, y )
        
    def test__mergeb(self):
        
        x = np.array( [ 3,4,1,2 ] )
        
        y = np.empty_like( x )
        sas._merge2( x, y, 0, 2, 4 )
        
        e = np.sort( x )
        np.testing.assert_array_equal( e, y )
        
    def test__mergeb2(self):
        
        x = np.array( [ 1, 3, 2, 4 ] )
        
        y = np.empty_like( x )
        sas._merge2( x, y, 0, 2, 4 )
        
        e = np.sort( x )
        np.testing.assert_array_equal( e, y ) 
        
    def test_merge(self):
        
        x = np.array( [ 1, 3, 2, 4, 5, 7, 6, 8, 9 ] )
        
        r = sas.merge( x )
        e = np.sort( x )
        np.testing.assert_equal( e, r )
        
    
        
    def test_merge_multi(self):
        
        n0 = 21
        n1 = 10000
        N  = 1
        
        for n in range( n0, n1 ):
            x = np.random.random_integers( 0, n, size=n )#.astype( np.int8 )
        
            with Timer(  ) as t0:
                for _ in range( N ):
                    r = sas.merge( x )
            r2 = x.copy()
            with Timer(  ) as t0b:                
                for _ in range( N ):
                    r2 = sas.merge2( r2 )
            r3 = x.copy()
            with Timer() as t_qs:
                sas.quick_sort(r3)
            with Timer(  ) as t1:
                for _ in range( N ):
                    e = np.sort( x, kind='merge' )
            np.testing.assert_equal( r, r2 )
            np.testing.assert_equal( r, e )
            print( 'nb/np performance %s'%(t0.interval/t1.interval ))
            print( 'nb2/np performance %s'%(t0b.interval/t1.interval ))
            print( 't_qs/np performance %s'%(t_qs.interval/t1.interval ))
            np.testing.assert_equal( e, r )
            
    def test_mergesort_recursive(self):
        
        x = np.array( [ 1, 3, 2, 4, 5, 7, 6, 8, 9 ] )
        
        r = sas.mergesort_recursive( x )
        e = np.sort( x )
        np.testing.assert_equal( e, r )