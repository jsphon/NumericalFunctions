import numpy as np
import unittest

import pyximport
pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
                              "include_dirs":np.get_include()},
                  reload_support=True)

import matplotlib.pyplot as plt
from numerical_functions.misc.timer import Timer
import numerical_functions.cython_funcs.math as cython_math
import numerical_functions.numba_funcs.math as numba_math


class Test(unittest.TestCase):
       
    def test_add_performance(self):
        
        num_tests = 100
        
        numba_timings = np.empty( num_tests )
        cython_timings = np.empty( num_tests )        
        numpy_timings = np.empty( num_tests )
               
        for j in range(num_tests):
            with Timer() as numba_timer:                
                numba_result = numba_math.add( 1, 2)
            
            with Timer() as cython_timer:                
                cython_result = cython_math.add( 1, 2 )        
        
            with Timer() as numpy_timer:                
                numpy_result = np.add( 1, 2 )
            
            np.testing.assert_equal( numpy_result, numba_result )
            np.testing.assert_equal( numpy_result, cython_result )
    
            numba_timings[ j ] = numba_timer.interval
            cython_timings[ j ] = cython_timer.interval
            numpy_timings[ j ] = numpy_timer.interval
        
        print( 'Best Cython Result : %s'%cython_timings.min() )    
        print( 'Best Numba Result : %s'%numba_timings.min() )        
        print( 'Best Numpy Result : %s'%numpy_timings.min() )
        
        y = [ cython_timings.min(), numba_timings.min(), numpy_timings.min() ]
        x = np.arange( len( y ) )
        plt.bar( x, y )
        plt.ylabel( 'Timings' )
        plt.xticks( x+0.5, [ 'Cython', 'Numba', 'Numpy' ] )
        plt.show()
                    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()