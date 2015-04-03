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

import numerical_functions.cython_funcs.indexing as indexing
import numerical_functions.numba_funcs.indexing as numba_indexing

from numerical_functions.misc.timer import Timer

class Test(unittest.TestCase):

    def test_take(self):
        
        x = np.linspace( 0, 1, 100 )
        idx = np.array( [ 3,2,1 ] )
        
        cresult = indexing.take( x, idx )
       
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()