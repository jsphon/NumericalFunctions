'''
Created on 9 Jul 2015

@author: jon
'''

from pprint import pprint 

import numpy as np
import pandas as pd
import json
import unittest
from numerical_functions.misc.timer import Timer        

from jagged_array.jagged_array import JaggedArray
from jagged_array import fns
        
class FnsTests( unittest.TestCase ):
    
    def setUp(self):
        
        price0 = [ 1, 2 ]
        price1 = [ 2, 3 ]
        price2 = [ 1, 2, 3 ]

        prices=[price0,price1,price2]
        
        self.prices = JaggedArray.from_list( prices )
        
        size0 = [ 1, 2 ]
        size1 = [ 3, 4 ]
        size2 = [ 5, 6, 7 ]
        
        sizes = [size0,size1,size2]
        self.sizes = JaggedArray.from_list( sizes )
        
    def test_cumsum(self):
        r = fns.to_frame( self.prices, self.sizes )
        print(r.cumsum())
        
    def test_to_frame(self):
        r = fns.to_frame( self.prices, self.sizes )
        print(r)
        
    def test_frame_to_jagged_array(self):
        
        data = [ [ 1, 2, 0 ],
                [ 1, 5, 4 ],
                [ 0, 11, 12 ] ]
        columns = [ 1, 2, 3 ]
        df = pd.DataFrame( data, columns=columns )
        
        keys, values = fns.frame_to_jagged_array(df)
        
        self.assertIsInstance( keys, JaggedArray )
        self.assertIsInstance( values, JaggedArray )
        
        np.testing.assert_equal( np.array( [ 1, 2 ] ), keys[0] )
        np.testing.assert_equal( np.array( [ 1, 2, 3 ] ), keys[1] )
        np.testing.assert_equal( np.array( [ 2, 3 ] ), keys[2] )
        
        np.testing.assert_equal( np.array( [ 1, 2 ] ), values[0] )
        np.testing.assert_equal( np.array( [ 1, 5, 4 ] ), values[1] )
        np.testing.assert_equal( np.array( [ 11, 12 ] ), values[2] )