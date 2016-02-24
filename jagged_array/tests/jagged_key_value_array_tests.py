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

        
class MoreJaggedKeyValueArrayTests( unittest.TestCase ):

    def test_to_dense_slice_from_beginning(self):
        ''' Check that it works when the bounds start before/after the end'''    
        k0 = [ 10, 11 ]
        k1 = [ 12, 13 ]
        k2 = [ 11, 12, 13, 14 ]
        
        v0 = [ 1, 2 ]
        v1 = [ 2, 3 ]
        v2 = [ 4, 5, 6, 7 ]
        
        keys = k0 + k1 + k2
        vals = v0 + v1 + v2

        bounds = [ 0, 2, 4, 7 ]
        
        arr = JaggedKeyValueArray( keys, vals, bounds )
        
        v, k = arr[1:].to_dense()
        
        expected_v = np.array( [ [ 0, 2, 3 ], [4, 5, 6 ] ] )
        expected_k = np.array( [ 11, 12, 13 ] )
 
        np.testing.assert_array_equal( expected_v, v )
        np.testing.assert_array_equal( expected_k, k )
        
    def test_to_dense_slice_to_end(self):
        ''' Check that it works when the bounds start before/after the end'''    
        k0 = [ 10, 11, 12 ]
        k1 = [ 12, 13 ]
        k2 = [ 11, 12, 13, 14 ]
        
        v0 = [ 1, 2, 3 ]
        v1 = [ 2, 3 ]
        v2 = [ 4, 5, 6, 7 ]
        
        keys = k0 + k1 + k2
        vals = v0 + v1 + v2

        bounds = [ 0, 3, 5, 8 ]
        
        arr = JaggedKeyValueArray( keys, vals, bounds )
        
        v, k = arr[:2].to_dense()
        
        expected_v = np.array( [ [ 1, 2, 3, 0 ], [0, 0, 2, 3 ] ] )
        expected_k = np.array( [ 10, 11, 12, 13 ] )
 
        np.testing.assert_array_equal( expected_v, v )
        np.testing.assert_array_equal( expected_k, k )
                                      
                                      
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
 
    def test___bool__(self):
        
        self.assertTrue( bool( self.arr ) )
        
    def test___bool__False(self):
        k = []
        v = []
        bounds = []
        arr = JaggedKeyValueArray( k, v, bounds )
        self.assertFalse( bool( arr ) )
        
    def test_to_dense_projection(self):
        projection = np.array( [ 10, 11, 12, 13, 14 ] )
        d= self.arr.to_dense_projection( projection )
        
        expected_d = np.array( [ [ 0, 1, 0, 0, 0 ],
                                 [ 0, 0, 2, 3, 0 ],
                                 [ 0, 4, 5, 6, 0 ] ] )
        
        print( 'd', d )
        np.testing.assert_array_equal( expected_d, d )
 
    def test_cumsum(self):
        
        dense, cols = self.arr.to_dense()
        cs          = dense.cumsum( axis=0 )
        e           = JaggedKeyValueArray.from_dense( cs, cols)
        
        print(e)
        r = self.arr.cumsum()
        
        np.testing.assert_array_equal( e.bounds, r.bounds )
        np.testing.assert_array_equal( e.keys, r.keys )
        np.testing.assert_array_equal( e.values, r.values )

 
    def test_from_lists(self):
        
        key_list = [ self.k0, self.k1, self.k2 ]
        val_list = [ self.v0, self.v1, self.v2 ]
        
        result = JaggedKeyValueArray.from_lists( key_list, val_list )
 
        self.assertEqual( self.arr, result )
 
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
        
        r = self.arr[ :2,-2: ]

        self.assertIsInstance( r, tuple )

        e0 = np.array( [[  0.,  11.],
             [ 12.,  13.],
             [ 12.,  13.]] )
        e1 = np.array( [[  0.,  1.],
             [ 2.,  3.],
             [ 5.,  6.]] )
        #print( r[1] )
        np.testing.assert_equal( e0, r[0] )
        np.testing.assert_equal( e1, r[1] )
        
    def test___getitem__2dslice4(self):
        
        k, v = self.arr[ :,-1 ]
        
        expected_keys = np.array( [ 11, 13, 13 ] )
        expected_vals = np.array( [ 1, 3, 6 ] )
        
        np.testing.assert_equal( expected_keys, k )
        np.testing.assert_equal( expected_vals, v )
    
    def test_to_dense(self):
        
        data, cols = self.arr.to_dense()
        print( data )
        print( cols )
        
    def test_to_dense2(self):
        keys = [ [ 0 ], [ 1, 2, 3 ], [2, 3 ] ]
        values = [ [ 10 ], [ 21, 22, 23 ], [ 32, 33 ] ]
        
        arr = JaggedKeyValueArray.from_lists( keys, values )
        
        data, cols = arr.to_dense()
        print( data )
        print( cols )
        
    def test_from_dense(self):
        data = [ [ 0, 1, 2 ], 
                 [ 3, 0, 4 ],
                 [ 0, 5, 0 ] ]
        cols = [ 10, 20, 30 ]
        
        data = np.array(data)
        cols = np.array(cols)
        r = JaggedKeyValueArray.from_dense(data, cols )
        print(r)
        
        e0 = np.array( [ 1, 2, 3, 4, 5 ] )
        e1 = np.array( [ 20, 30, 10, 30, 20 ] )
        e2 = np.array( [ 0, 2, 4, 5 ] )
        np.testing.assert_array_equal( e0, r.values )
        np.testing.assert_array_equal( e1, r.keys )
        np.testing.assert_array_equal( e2, r.bounds )
        
    def test_from_dense_nb(self):
        data = [ [ 0, 1, 2 ], 
                 [ 3, 0, 4 ],
                 [ 0, 5, 0 ] ]
        cols = [ 10, 20, 30 ]
        
        data = np.array(data)
        cols = np.array(cols)
        r = JaggedKeyValueArray.from_dense_nb(data, cols )
        print(r)
        
        e0 = np.array( [ 1, 2, 3, 4, 5 ] )
        e1 = np.array( [ 20, 30, 10, 30, 20 ] )
        e2 = np.array( [ 0, 2, 4, 5 ] )
        np.testing.assert_array_equal( e0, r.values )
        np.testing.assert_array_equal( e1, r.keys )
        np.testing.assert_array_equal( e2, r.bounds )
        