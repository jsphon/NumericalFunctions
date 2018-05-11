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
import jagged_array.jagged_array as mod
import jagged_array.fns as jf

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
        np.testing.assert_equal( np.array([2,3]),r[1])
        
    def test___getitem__slice_2d_1(self):
        """ slice, int """
        r = self.ja[:2,-1]
        self.assertIsInstance( r, np.ndarray )
        self.assertEqual( 2, len(r) )
        np.testing.assert_equal( np.array([1,3]),r )
        
    def test___getitem__slice_2d_2(self):
        """ slice, int """
        r = self.ja[:,-1]
        self.assertIsInstance( r, np.ndarray )
        self.assertEqual( 4, len(r) )
        np.testing.assert_equal( np.array([1,3,6,10]),r )
        print(r)
        
    def xxxtest___getitem__slice_2d_3(self):
        """ Double slice """
        r = self.ja[:,-2:-1]
        self.assertIsInstance( r, np.ndarray )
        self.assertEqual( ( 4, 2 ), r.shape )
        np.testing.assert_equal( np.array([1,3,6,10]),r )
        print(r)

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


    def test_kv_to_dense_timer(self):
        
        k0 = [ 1, ]
        k1 = [ 2, 3 ]
        k2 = [ 1, 3, 4 ]   
        k3 = [ 1,2,3,4,5 ]     

        kdata  = np.array( k0 + k1 + k2 + k3, dtype=np.int )
        bounds = np.array( [ 0, 1, 3, 6, 11 ] )
        
        v0 = [ 10 ]
        v1 = [ 20, 30 ]
        v2 = [ 11, 12, 13 ]
        v3 = [ 1, 2, 3, 4, 5 ]
        vdata = np.array( v0 + v1 + v2 + v3, dtype=np.int )
        
        r = mod.kv_to_dense( kdata, vdata, bounds )
        

        N = 1000
        with Timer( 'nb' ) as t0:
            for _ in range( N ):
                r = mod.kv_to_dense( kdata, vdata, bounds )
                
        k = JaggedArray( kdata, bounds )
        v = JaggedArray( vdata, bounds )
        with Timer( 'py' ) as t1:
            for _ in range( N ):
                jf.to_frame( k, v, 0 )

        r = t0.interval / t1.interval
        print( 'ratio : %s'%r )