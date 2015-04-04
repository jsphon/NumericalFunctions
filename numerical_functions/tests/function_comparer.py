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

    def compare_performance(self, title, fns, arg_gen_fn, sizes, num_tests=100, assert_results=True, assert_args=None ):

        timings = {}
        for nm in fns:
            timings[ nm ] = np.empty( ( sizes.shape[0], num_tests ) )
        
        fn_names = list( fns.keys() )
        
        for i, size in enumerate( sizes ):
            
            print( '%s : Analysing size %s of %s : %s'%(title, i+1, len( sizes ), size ) )
        
            args = arg_gen_fn( size )
        
            for j in range( num_tests ):
                
                results={}
                for nm, fn in fns.items():
                    with Timer() as timer:
                        results[nm] = fn( *args )
                    timings[ nm ][ i, j ] = timer.interval
                    
                if assert_results:
                    result0 = results[ fn_names[0] ]
                    for fn_name in fn_names[1:]:
                        result1 = results[ fn_name ]
                        if ( result0 is not None ) and ( result1 is not None ):
                            np.testing.assert_array_almost_equal( result0, result1 )
                            
                if assert_args is not None:
                    # Do this later
                    pass
        
        for nm in fns:
            plt.plot( sizes, timings[ nm ].min( axis=1 ), label=nm )
        plt.title( '%s Performance Test'%title )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.ion()
        plt.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()