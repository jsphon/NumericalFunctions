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

from cython_funcs.portfolio_math import portfolio_s2_by_index

from numba_funcs.timer import Timer

class Test(unittest.TestCase):

    def test_portfolio_s2(self):
        X = np.random.randn(100,10)
        cv = np.cov( X, rowvar=0 )
        w  = np.ones( cv.shape[0] )
        s2_cython = pm.portfolio_s2( cv, w )
        print( s2_cython )
        
        s2_numba = helpers.portfolio_s2( cv, w )
        print( s2_numba )
        
        np.testing.assert_almost_equal( s2_numba, s2_cython )       
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()