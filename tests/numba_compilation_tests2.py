'''
Created on 30 Oct 2014

@author: Jon
'''


import numpy as np

from numba import double, jit, autojit, void, int_
from numba.types import pyobject

import time

class Timer:    
    
    def __init__(self,title=None):
        self.title=title
        
    def __enter__(self):
        if self.title:
            print( 'Beginning {0}'.format( self.title ) )
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        if self.title:
            print( '{1} took {0:0.3f} seconds'.format( self.interval, self.title ) )
        else:
            print( 'Timer took {0:0.3f} seconds'.format( self.interval ) )
     
def _find_min_max( odds, comp_ids, ngroups ):

    r_min = np.ndarray( ngroups, dtype=np.float ) 
    r_max = np.ndarray( ngroups, dtype=np.float )
    
    r_min[:] = np.finfo( np.float ).max
    r_max[:] = np.finfo( np.float ).min    
    
    r_argmin = np.ndarray( ngroups, dtype=np.int )
    r_argmax = np.ndarray( ngroups, dtype=np.int )
    
    r_argmin[:]=np.nan
    r_argmax[:]=np.nan
    
    for i in range( len( comp_ids ) ):
        cid  = comp_ids[ i ]
        o    = odds[ i ]
    
        if o<r_min[cid]:
            r_argmin[cid] = i
            r_min[cid] = o
            
        if o>r_max[cid]:
            r_argmax[cid]=i
            r_max[cid]=o
            
    return r_argmin, r_min, r_argmax, r_max

odds_values = np.random.randn(100)
comp_ids    = np.repeat( np.arange(10), 10 )
ngroups     = 10
    
with Timer( 'jitting' ):
    f = jit(pyobject( double[:], int_[:], int_ ) )( _find_min_max )
     
with Timer( 'calculating' ):
    r = f( odds_values, comp_ids, ngroups )
    
with Timer( 'calculating' ):
    r = f( odds_values, comp_ids, ngroups )    