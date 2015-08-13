'''
Created on 9 Jul 2015

@author: jon
'''

from pprint import pprint 

import numpy as np
import json
import unittest
from numerical_functions.misc.timer import Timer        

from jagged_array.jagged_key_value_array import JaggedKeyValueArray

        
class JaggedKeyValueArrayTests( unittest.TestCase ):
    
    def setUp(self):
        
        self.k0 = [ 11, ]
        self.k1 = [ 12, 13 ]
        self.k2 = [ 11, 12, 13 ]
        
        self.v0 = [ 1, ]
        self.v1 = [ 2, 3 ]
        self.v2 = [ 4, 5, 6 ]
        
        self.keys = self.k0 + self.k1 + self.k2
        self.vals = self.v0 + self.v1 + self.v2

        self.bounds = [ 0, 1, 3, 6 ]
        
        self.arr = JaggedKeyValueArray( self.keys, self.vals, self.bounds )
 
    def test___len__(self):
        self.assertEqual( 3, len( self.arr ) )
        
    def test___getitem__row0(self):
        
        r = self.arr[0]
        
        e0 = np.array( self.k0 )
        e1 = np.array( self.v0 )
        
        np.testing.assert_array_equal( e0, r[0] )
        np.testing.assert_array_equal( e1, r[1] )
        
    def test___getitem__row1(self):
        
        r = self.arr[1]
        
        e0 = np.array( self.k1 )
        e1 = np.array( self.v1 )
        
        np.testing.assert_array_equal( e0, r[0] )
        np.testing.assert_array_equal( e1, r[1] )
        
    def test__getitem_1dslice1(self):
        
        r = self.arr[:1]
        print( 'test__getitem_1dslice: %s'%r )
        self.assertIsInstance( r, JaggedKeyValueArray )
        self.assertEqual( 1, len( r ) )
        self.assertEqual( 0, r.bounds[0] )
        self.assertEqual( 1, r.bounds[1] )

    def test__getitem_1dslice2(self):
        
        r = self.arr[1:3]
        print( 'test__getitem_1dslice: %s'%r )
        self.assertIsInstance( r, JaggedKeyValueArray )
        self.assertEqual( 2, len( r ) )
        self.assertEqual( 1, r.bounds[0] )
        self.assertEqual( 3, r.bounds[1] )
        self.assertEqual( 6, r.bounds[2] )
        
    def test___getitem__2dslice(self):
        
        k, v = self.arr[0,0]
        self.assertEqual( 11, k )
        self.assertEqual( 1, v )
        
        k, v = self.arr[ 1, 0 ]
        self.assertEqual( 12, k )
        self.assertEqual( 2, v )
        
        k, v = self.arr[ 1, -1 ]
        self.assertEqual( 13, k )
        self.assertEqual( 3, v )
        
    def test___getitem__2dslice2(self):
        
        k, v = self.arr[ 0,:2 ]
        
        self.assertEqual( self.k0, k )
        self.assertEqual( self.v0, v )
        
        k, v = self.arr[ 1,:1 ]
        
        self.assertEqual( self.k1[:1], k )
        self.assertEqual( self.v1[:1], v )
        
        k, v = self.arr[ 2, 1: ]
        
        np.testing.assert_array_equal( self.k2[1:], k )
        np.testing.assert_array_equal( self.v2[1:], v )
        
    def test___getitem__2dslice3(self):
        
        r = self.arr[ :2,:2 ]
        
        self.assertIsInstance( r, JaggedKeyValueArray )

        
        
"""
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
        
    def test_kv_to_dense(self):
        
        k0 = [ 1, ]
        k1 = [ 2, 3 ]
        k2 = [ 1, 3, 4 ]        

        kdata  = np.array( k0 + k1 + k2, dtype=np.int )
        bounds = np.array( [ 0, 1, 3, 6 ] )
        
        v0 = [ 10 ]
        v1 = [ 20, 30 ]
        v2 = [ 11, 12, 13 ]
        vdata = np.array( v0 + v1 + v2, dtype=np.int )
        
        r = mod.kv_to_dense( kdata, vdata, bounds )
        print( r )
        
        print( 'cumsum...' )
        print( np.cumsum( r[0] ) )
        
        expected0 = [ [ 10, 0, 0, 0 ],
                      [ 0, 20, 30, 0 ],
                      [ 11, 0, 12, 13 ] ]
        expected0 = np.array( expected0 )
        
        expected1 = np.array( [ 1,2,3,4 ])
        
        np.testing.assert_equal( expected0, r[0] )
        np.testing.assert_equal( expected1, r[1] )

"""