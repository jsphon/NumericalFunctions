'''
Created on 14 Nov 2014

@author: Jon
'''


import my_numba
from timer import Timer

import os
import numpy as np
import pandas as pd
import unittest


class Test(unittest.TestCase):

    def test_take( self ):
        
        N = 100000
        m = 100
        idx = np.random.random_integers( 0, N, m )
        x = np.random.randn( N )
        
        r = my_numba.take( x, idx )
        print(r)
        print( x.take( idx ) )
        
        self.assertEqual( m, r.shape[0] )
        self.assertTrue( np.all( r==x.take( idx ) ) )
        
    def test_take_performance( self ):
        
        N = 100000
        m = 100
        idx = np.random.random_integers( 0, N, m )
        x = np.random.randn( N )

        num_tests=10000

        with Timer( 'take' ):
            for i in range( num_tests ):    
                r0 = my_numba.take( x, idx )
                
        with Timer( 'Numba take' ):
            for i in range( num_tests ):    
                r1 = my_numba.jtake( x, idx )                
            
        with Timer( 'Numpy.take' ):
            for i in range( num_tests ):
                r2 = x.take( idx )
        
