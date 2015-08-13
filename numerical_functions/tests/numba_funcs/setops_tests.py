
import numpy as np
import unittest
import numerical_functions.numba_funcs.setops as mod

class Test(unittest.TestCase):


    def test_unique(self):
        
        x = np.array( [1, 2, 2, 3, 3, 3 ])
        r = mod.unique(x)
        
        print( r )

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()