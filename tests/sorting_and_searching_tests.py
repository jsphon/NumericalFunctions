'''
Created on 27 Mar 2015

@author: Jon
'''

import numba_funcs as nf
import numba_funcs.array_reductions as ar
import numba_funcs.sorting_and_searching as sas
from numba_funcs.timer import Timer
import numpy as np
import unittest


class Test(unittest.TestCase):

    def test_binary_search(self):
        for _ in range( 100 ):
            x = np.random.random_integers( 0, 100, 10 )
            x.sort()    
            
            r = sas.binary_search( x, x[5] )
            expected = np.searchsorted( x, x[5] )
            
            self.assertEqual( expected, r )
            
    def test_binary_search_performance(self):
        
        x = np.random.random_integers( 0, 100000, 10000 )
        x.sort()    
            
        sas.binary_search( x, x[500] )
        
        num_tests = 1000
        
        with Timer( 'nb' ):
            for _ in range( num_tests ):            
                sas.binary_search( x, x[5000] )
                
        with Timer( 'np' ):
            for _ in range( num_tests ):
                np.searchsorted( x, x[5000] )
                
    def test_quick_sort(self):
        x = np.random.random_integers( 0, 100000, 10000 )
        
        self.assertFalse( ar.is_monotonic_increasing( x ) )
        sas.quick_sort( x )        
        self.assertTrue( ar.is_monotonic_increasing( x ) )
        
    def test_quick_sort_performance(self):
        x = np.random.random_integers( 0, 100000, 10000 )
        
        x0 = x.copy()
        x1 = x.copy()
        
        sas.quick_sort( x0.copy() ) 
        with Timer( 'nb' ):
            sas.quick_sort( x0 )

        with Timer( 'np' ):
            np.sort( x1 )