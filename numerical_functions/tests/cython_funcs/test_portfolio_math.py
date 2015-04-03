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

import cython_funcs.portfolio_math as pm
import src.helpers as helpers

from cython_funcs.portfolio_math import portfolio_s2 as portfolio_s2_cython
from cython_funcs.portfolio_math import portfolio_s2_call as portfolio_s2_call_cython
from cython_funcs.portfolio_math import portfolio_s2_opt as portfolio_s2_opt_cython
from src.helpers import portfolio_s2 as portfolio_s2_numba
from src.helpers import portfolio_s2_call as portfolio_s2_call_numba

from numba_funcs.timer import Timer

class Test(unittest.TestCase):
    
    def setUp(self):
        N = 1000
        X = np.random.randn( N, 100 )
        self.cv = np.cov( X, rowvar=0 )
        self.w  = np.ones( self.cv.shape[0] ) / self.cv.shape[0]

    def test_portfolio_s2(self):
        X = np.random.randn(100,10)
        cv = np.cov( X, rowvar=0 )
        w  = np.ones( cv.shape[0] )
        s2_cython = pm.portfolio_s2( cv, w )
        print( s2_cython )
        
        s2_numba = helpers.portfolio_s2( cv, w )
        print( s2_numba )
        
        np.testing.assert_almost_equal( s2_numba, s2_cython )
        
    def test_portfolio_s2_performance(self):
        X = np.random.randn(100,10)
        cv = np.cov( X, rowvar=0 )
        w  = np.ones( cv.shape[0] )
        
        num_tests=1000
        
        portfolio_s2( cv, w )
        with Timer( 'Cython' ) as cython_timer:
            for _ in range( num_tests ):
                s2_cython = pm.portfolio_s2( cv, w )
        
        helpers.portfolio_s2( cv, w )
        with Timer( 'Numba' ) as numba_timer:
            for _ in range( num_tests ):
                s2_numba = helpers.portfolio_s2( cv, w )
        
        np.testing.assert_almost_equal( s2_numba, s2_cython )
        
        ratio = cython_timer.interval / numba_timer.interval
        print( 'Cython took %s as long as Numba.'%ratio )
        
    def test_portfolio_s2_performance_by_size(self):
        
        sizes = [ 2, 3, 4, 6, 8, 12, 16, 32, 48, 64, 96, 128, 196, 256 ]
        cython_opt_timings = []
        cython_timings = []
        numba_timings = []
        for size in sizes:
            X = np.random.randn(100,size)
            cv = np.cov( X, rowvar=0 )
            w  = np.ones( cv.shape[0] )
            
            num_tests=1000
            
            with Timer( 'portfolio_s2_opt_cython' ) as cython_opt_timer:
                for _ in range( num_tests ):
                    s2_cython = portfolio_s2_opt_cython( cv, w )
            cython_opt_timings.append( cython_opt_timer.interval )
            
            with Timer( 'portfolio_s2_cython' ) as cython_timer:
                for _ in range( num_tests ):
                    s2_cython = portfolio_s2_cython( cv, w )
            cython_timings.append( cython_timer.interval )
            
            portfolio_s2_numba( cv, w )
            with Timer( 'portfolio_s2_numba' ) as numba_timer:
                for _ in range( num_tests ):
                    s2_numba = portfolio_s2_numba( cv, w )
            numba_timings.append( numba_timer.interval )
            
        plt.plot( sizes, cython_timings, label='Cython' )
        plt.plot( sizes, cython_opt_timings, label='Cython Opt' )        
        plt.plot( sizes, numba_timings, label='Numba' )
        plt.title( 'Execution Time By Covariance Size' )
        plt.legend()
        plt.show()
            
            
    def test_portfolio_s2_call_performance_by_size(self):
        
        sizes = [ 2, 3, 4, 6, 8, 12, 16, 32, 48, 64, 96, 128, 196, 256 ]
        cython_timings = []
        numba_timings = []
        for size in sizes:
            X = np.random.randn(100,size)
            cv = np.cov( X, rowvar=0 )
            w  = np.ones( cv.shape[0] )
            
            num_tests=1000
                       
            with Timer( 'portfolio_s2_cython' ) as cython_timer:
                for _ in range( num_tests ):
                    s2_cython = portfolio_s2_call_cython( cv, w )
            cython_timings.append( cython_timer.interval )
            
            portfolio_s2_call_numba( cv, w )
            with Timer( 'portfolio_s2_numba' ) as numba_timer:
                for _ in range( num_tests ):
                    s2_numba = portfolio_s2_call_numba( cv, w )
            numba_timings.append( numba_timer.interval )
            
        plt.plot( sizes, cython_timings, label='Cython' )  
        plt.plot( sizes, numba_timings, label='Numba' )
        plt.title( 'Function Call Time By Covariance Size' )
        plt.legend()
        plt.show()
        
    def test_pportfolio_s2_by_index(self):
        
        idx = np.arange( 10 )
        np.random.shuffle( idx )
        idx = idx[:5]
        
        s2 = pm.pportfolio_s2_by_index( self.cv, self.w, idx )
        
        cv2 = nf.square_take( self.cv, idx )
        expected_s2 = helpers.portfolio_s2( cv2, self.w.take( idx ) )
        
        self.assertEqual( expected_s2, s2 )
       
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()