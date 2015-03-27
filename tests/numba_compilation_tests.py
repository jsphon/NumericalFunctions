'''
Created on 30 Oct 2014

@author: Jon
'''

import numexpr as ne
import numpy as np
import pandas as pd
from timer import Timer


from my_numba.multi_thread_vectorizer import mvectorize
from numba import double, jit, autojit

from timer import Timer
odds_values = np.random.randn(100)
comp_ids    = np.repeat( np.arange(10), 10 )
ngroups     = ngroups=10
     
#with Timer( 'find_min_max' ):
#    r_argmin, r_min, r_argmax, r_max = find_min_max( odds_values, comp_ids, ngroups )

def f(x):
    return x

@jit(double(double))
def j_f(x):
    return x,x



print( f(0) )
with Timer( 'jf' ):
    print( j_f(0) )