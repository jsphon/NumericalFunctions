'''
Created on 27 Mar 2015

@author: Jon
'''

import matplotlib.pyplot as plt
from numba_funcs.timer import Timer
import numerical_functions.numba_funcs.indexing as indexing
import numpy as np
import unittest


class Test(unittest.TestCase):
        
    def test_take(self):
        x = np.linspace( 0, 100 )
        idx = np.random.random_integers( 0, 50, 20 )
        result = indexing.take( x, idx )
        expected = np.take( x, idx )
        np.testing.assert_array_equal( expected, result )
        
    def test_take_comparison(self):
        x = np.arange( 1e6 )
        idx = np.random.random_integers( 0, 1e5, 1e6 )
        
        indexing.take( x, idx )
        np.take( x, idx )
        
        with Timer( 'numba' ) as nbtimer:
            indexing.take( x, idx )
            
        with Timer( 'numpy' ) as nptimer:
            np.take( x, idx )
           
        ratio = nbtimer.interval / nptimer.interval
        print( 'numba version of take took %0.2f as long as numpy'%ratio) 
        
    
    def test_square_take(self):

        X = np.random.random_integers( 0, 50, 25 ).reshape( 5, 5 )
        idx = np.arange( 0, 4, 2 )
        result = np.empty( ( idx.shape[0], idx.shape[0] ) )
        indexing.square_take_to_out( X, idx, result )
        print( result )
        
        expected = X.take( idx, axis=0 ).take( idx, axis=1 )
        print( expected )
        
        np.testing.assert_array_equal( expected, result )
    
    def test_square_take_to_out(self):
        X = np.arange(25).reshape(5,5)
        idx = np.arange( 0, 4, 2 )
        result = np.empty( ( idx.shape[0], idx.shape[0] ) )
        indexing.square_take_to_out( X, idx, result )
        print( result )
        
        expected = X.take( idx, axis=0 ).take( idx, axis=1 )
        print( expected )
        
        np.testing.assert_array_equal( expected, result )
        
    def test_square_take_performance(self):
        X = np.arange(25).reshape(5,5)
        idx = np.arange( 0, 4, 2 )
        result = np.empty( ( idx.shape[0], idx.shape[0] ) )
        indexing.square_take_to_out( X, idx, result )
        
        result2 = indexing.square_take( X, idx )
        
        np.testing.assert_array_equal( result, result2 )

        num_tests = 1000
        
        nbts = []
        nbts2 = []
        npts = []        
        
        ms = ( 10, 20, 40, 80, 160 )#, 320, 640  )
        for m in ms:
            X = np.arange(m*m).reshape(m,m)
            idx = np.random.random_integers( 0, m-1, m//2 )
            result = np.empty( ( idx.shape[0], idx.shape[0] ) )
            with Timer( 'numba' ) as nbt:
                for _ in range( num_tests ):
                    indexing.square_take_to_out( X, idx, result )
            nbts.append( nbt.interval )   
            
            with Timer( 'numba2' ) as nbt:
                for _ in range( num_tests ):
                    r=indexing.square_take( X, idx ) 
            nbts2.append( nbt.interval ) 
            
            with Timer( 'numpy') as npt:
                for _ in range(num_tests):
                    X.take( idx, axis=0 ).take( idx, axis=1 )
            npts.append( npt.interval )   
            
        plt.plot( ms, nbts, label='nb to out' )
        plt.plot( ms, nbts2, label='nb new result')
        plt.plot( ms, npts, label='np' )
        plt.title( 'square_take_to_out performance test')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
            
    def test_square_and_rect_take_to_out(self):
        
        X = np.arange( 100 ).reshape( (10, 10 ) )
        idx0 = np.arange( 0, 4, 2 )
        idx1 = np.arange( 4, 6 )
        
        result = np.empty( ( idx0.shape[0], idx0.shape[0]+idx1.shape[0] ) )
        af.square_and_rect_take_to_out( X, idx0, idx1, result )
        
        print( X )
        print( idx0 )
        print( idx1 )
        print( result )
        
        np.testing.assert_array_equal( result[:,:2], af.square_take( X, idx0 ) )
        r2 = np.array( [ [ 4, 5 ], [24, 25 ] ] )
        np.testing.assert_array_equal( r2, result[:,2:])       

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()