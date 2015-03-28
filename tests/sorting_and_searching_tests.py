'''
Created on 27 Mar 2015

@author: Jon
'''

import numba_funcs as nf
from numba_funcs.timer import Timer
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