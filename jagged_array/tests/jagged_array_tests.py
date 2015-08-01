'''
Created on 9 Jul 2015

@author: jon
'''

from pprint import pprint 

import numpy as np
import json
import unittest
from numerical_functions.misc.timer import Timer        

from jagged_array.jagged_array import JaggedArray

import tempfile
        
class JaggedArrayTests( unittest.TestCase ):
    
    def setUp(self):
        
        x0 = [ 1, ]
        x1 = [ 2, 3 ]
        x2 = [ 4, 5, 6 ]
        x3 = [1,2,3,4,5,6,7,8,9,10]

        self.data = x0+x1+x2+x3
        self.bounds = [ 0, 1, 3, 6,16 ]
        
        self.ja = JaggedArray( self.data, self.bounds )
        
    def test_from_list(self):
        
        lst = [ [ 0, ], [ 1, 2, ], [ 3, 4, 5,  ]]
        r = JaggedArray.from_list( lst )
        print(r)
        self.assertIsInstance( r, JaggedArray )
        
    def test___eq__(self):
        self.assertEqual( self.ja, self.ja )
        
    def test___init__(self):
        print( 'Created jagged array:\n%s'%self.ja)
        
    def test___getitem___int(self):
        
        r = self.ja[0]        
        self.assertEqual( np.array([1]), r )
        
        r = self.ja[1]
        np.testing.assert_equal( np.array([2,3]),r)

        r = self.ja[2]        
        np.testing.assert_equal( np.array([4,5,6]),r)

    def test___getitem__slice(self):
        r = self.ja[:2]
        self.assertIsInstance( r, JaggedArray )
        self.assertEqual( 2, len(r) )
        self.assertEqual( np.array([1]), r[0] )
        np.testing.assert_equal( np.array([2,3]),self.ja[1])

    def test___len__(self):
        self.assertEqual( 4, len( self.ja ) )
        
    def test__iter__(self):
        
        for i, row in enumerate( self.ja ):
            self.assertIsInstance( row, np.ndarray )
            np.testing.assert_equal( np.array(self.ja[i]),row)
            print(row)
            
    def test_save(self):
        fn = '/tmp/ja.npz'
        self.ja.save(fn)
        
    def test_load(self):
        fn = '/tmp/ja.npz'        
        self.ja.save(fn)
        
        ja2=JaggedArray.load(fn)
        print(ja2)
        self.assertEqual( self.ja, ja2 )
