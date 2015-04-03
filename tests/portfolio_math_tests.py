import numpy as np
import unittest

import pyximport
pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
                              "include_dirs":np.get_include()},
                  reload_support=True)

import matplotlib.pyplot as plt
from numerical_functions.misc.timer import Timer
import numerical_functions.cython_funcs.portfolio_math as cython_math
import numerical_functions.numba_funcs.portfolio_math as numba_math
import numerical_functions.numpy_funcs.portfolio_math as numpy_math

class Test(unittest.TestCase):

    def setUp(self):
        self.cv = makeRandomCovarianceMatrix( 10 )
        self.w  = np.ones( self.cv.shape[0], dtype=np.int ) / self.cv.shape[0]

    def test_portfolio_var(self):
        w = np.ones( self.cv.shape[0] ) / self.cv.shape[0]
        
        cython_var = cython_math.pportfolio_var( self.cv, w )
        numba_var = numba_math.portfolio_var( self.cv, w )
        numpy_var = numpy_math.portfolio_var( self.cv, w )

        np.testing.assert_array_almost_equal( numpy_var, cython_var )
        np.testing.assert_array_almost_equal( numpy_var, numba_var )
        
    def test_portfolio_var_performance(self):
        
        sizes = 2**np.arange(1,10)
        
        numba_timings = np.empty( sizes.shape[0] )
        cython_timings = np.empty( sizes.shape[0] )
        numpy_timings = np.empty( sizes.shape[0] )
        num_tests = 100        
        for i, dsize in enumerate( sizes ):
            cv = makeRandomCovarianceMatrix( dsize )        
            w  = np.ones( dsize, dtype=np.int ) / dsize
        
            numba_math.portfolio_var( cv, w )
            with Timer( 'Numba' ) as numba_timer:
                for _ in range( num_tests ):
                    numba_result = numba_math.portfolio_var( cv, w )
                
            with Timer( 'Cython' ) as cython_timer:
                for _ in range( num_tests ):
                    cython_result = cython_math.pportfolio_var( cv, w )
                
            with Timer( 'Numpy' ) as numpy_timer:
                for _ in range( num_tests ):
                    numpy_result = numpy_math.portfolio_var( cv, w )
                
            np.testing.assert_array_almost_equal( numpy_result, numba_result )
            np.testing.assert_array_almost_equal( numpy_result, cython_result )
        
            numba_timings[ i ] = numba_timer.interval
            cython_timings[ i ] = cython_timer.interval
            numpy_timings[ i ] = numpy_timer.interval
            
        plt.plot( sizes, numba_timings, label='Numba' )
        plt.plot( sizes, cython_timings, label='Cython')
        plt.plot( sizes, numpy_timings, label='Numpy' )
        plt.title( 'Square Take() Performance Test')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

def makeRandomCovarianceMatrix( dsize ):
    N = 2*dsize
    X = np.random.randn( N, dsize )
    return np.cov( X, rowvar=0 )


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()