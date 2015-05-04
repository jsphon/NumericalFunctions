'''
Created on 1 May 2015

@author: jon
'''
import unittest
from numerical_functions.numba_funcs.date import bdate_range_slicer
from numerical_functions.misc.timer import Timer, AccumulatedTimer
import numpy as np
import pandas as pd

class Test(unittest.TestCase):


    def test_bdate_range_slicer(self):

        fd = pd.datetime( 2010,1,1 )
        ld = pd.datetime( 2012,1,1 )
        
        bdate_range = bdate_range_slicer()

        with Timer( 'my bdate_range' ):
            dr_slice = bdate_range( fd, ld )
        with Timer( 'bdate_range' ):
            dr_pd = pd.bdate_range( fd, ld )
            
        np.testing.assert_array_equal( dr_pd, dr_slice )
        
    def test_bdate_range_slicer2(self):

        bdate_range = bdate_range_slicer()
        dr = pd.bdate_range( pd.datetime( 1990,1,1 ), pd.datetime.today() )
        
        # JIT it
        bdate_range( dr[0], dr[-1] )
        
        slice_timer = AccumulatedTimer()
        pd_timer = AccumulatedTimer()
        for _ in range(100):

            np.random.shuffle( dr.values )
            fd = dr[0]
            ld = dr[1]     
            if fd>ld:
                fd,ld=ld,fd
            
            print( 'Testing bdate_range_slicer from %s to %s'%(fd,ld))       
    
            with slice_timer:
                dr_slice = bdate_range( fd, ld )
            with pd_timer:
                dr_pd = pd.bdate_range( fd, ld )
                
            np.testing.assert_array_equal( dr_pd, dr_slice )
        
        print( 'Slicing the bdate_range took %s'%slice_timer )
        print( 'pd.bdate_range took %s'%pd_timer )
        print( 'slicer / pd = %s'%(slice_timer.interval / pd_timer.interval))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()