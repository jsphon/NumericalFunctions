'''
Created on 3 Apr 2015

@author: Jon
'''

import matplotlib.pyplot as plt
import numpy as np
import unittest
from numerical_functions.misc.timer import Timer

class FunctionComparer(unittest.TestCase):
    """ Test class for comparing function performance """

    def compare_performance(self, title, fns, arg_gen_fn, sizes, num_tests=100 ):

        timings = {}
        for nm in fns:
            timings[ nm ] = np.empty( ( sizes.shape[0], num_tests ) )
        
        fn_names = list( fns.keys() )
        
        for i, size in enumerate( sizes ):
            
            print( 'Analysing size %s of %s'%(i+1, 1+len( sizes ) ) )
        
            args = arg_gen_fn( size )
        
            for j in range( num_tests ):
                
                results={}
                for nm, fn in fns.items():
                    with Timer() as timer:
                        results[nm] = fn( *args )
                    timings[ nm ][ i, j ] = timer.interval
                    
                result0 = results[ fn_names[0] ]
                for fn_name in fn_names[1:]:
                    result1 = results[ fn_name ]
                    np.testing.assert_array_almost_equal( result0, result1 )
        
        for nm in fns:
            plt.plot( sizes, timings[ nm ].min( axis=1 ), label=nm )
        plt.title( '%s Performance Test'%title )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()