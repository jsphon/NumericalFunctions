# -*- coding: utf-8 -*-

from numba import vectorize
from timeit import repeat

import numpy as np
from numba import jit, void, double

from my_numba.multi_thread_vectorizer import mvectorize

if __name__=='__main__':
    """ Example Usage """
    
    def looped_floor_closest_valid_odds( x ):
    
        r = np.zeros_like( x )
    
        for i in range( len( r ) ):
            xi = x[i]
            if xi<=1.0:
                r[i] = np.nan
            elif xi<=2.0:            
                r[i] = 0.01 * np.floor( xi / 0.01 )            
            elif xi<=3.0:                        
                r[i] = 0.02 * np.floor( xi / 0.02 )            
            elif xi<=4.0:            
                r[i] = 0.05 * np.floor( xi / 0.05 )
            elif xi<=6.0:            
                r[i] = 0.1 * np.floor( xi / 0.1 )
            elif xi<=10.0:            
                r[i] = 0.5 * np.floor( xi / 0.5 )
            elif xi<=20.0:            
                r[i] = 1.0 * np.floor( xi / 1.0 )
            elif xi<=30.0:            
                r[i] = 2.0 * np.floor( xi / 2.0 )
            elif xi<=50.0:            
                r[i] = 2.0 * np.floor( xi / 2.0 )
            elif xi<=100.0:            
                r[i] = 5.0 * np.floor( xi / 5.0 )
            elif xi<=1000.0:            
                r[i] = 10.0 * np.floor( xi / 10.0 )
            else:            
                r[i] = 1000.0
        return r
    
    """
    This was the original fastest function
    Cannot use jit( ..., nopython=True ) as np.zeros_like is currently not compatible in no python mode
    """
    signature = double[:](double[:])
    lf        = jit( signature )( looped_floor_closest_valid_odds )
    
    def floor_closest_valid_odds( xi ):

        if xi<=1.0:
            return 1.0
        elif xi<=2.0:            
            return 0.01 * np.floor( xi / 0.01 )            
        elif xi<=3.0:                        
            return 0.02 * np.floor( xi / 0.02 )            
        elif xi<=4.0:            
            return 0.05 * np.floor( xi / 0.05 )
        elif xi<=6.0:            
            return 0.1 * np.floor( xi / 0.1 )
        elif xi<=10.0:            
            return 0.5 * np.floor( xi / 0.5 )
        elif xi<=20.0:            
            return 1.0 * np.floor( xi / 1.0 )
        elif xi<=30.0:            
            return 2.0 * np.floor( xi / 2.0 )
        elif xi<=50.0:            
            return 2.0 * np.floor( xi / 2.0 )
        elif xi<=100.0:            
            return 5.0 * np.floor( xi / 5.0 )
        elif xi<=1000.0:            
            return 10.0 * np.floor( xi / 10.0 )
        else:            
            return 1000.0
        return 0.0

    signature = double(double,)
    print( 'Compiling jit function' )
    nb_floor_closest_valid_odds = jit(signature, nopython=True)(floor_closest_valid_odds)
    
    print( 'Compiling 4 thread')
    mf4 = mvectorize( nb_floor_closest_valid_odds, ( double[:], double[:] ), num_threads=4 )
    print( 'Compiling 6 thread')
    mf6 = mvectorize( nb_floor_closest_valid_odds, ( double[:], double[:] ), num_threads=6 )    
    print( 'Compiling 7 thread')
    mf7 = mvectorize( nb_floor_closest_valid_odds, ( double[:], double[:] ), num_threads=7 )    
    print( 'Compiling 8 thread')
    mf8 = mvectorize( nb_floor_closest_valid_odds, ( double[:], double[:] ), num_threads=8 )
    print( 'Compiling 16 thread')
    mf16 = mvectorize( nb_floor_closest_valid_odds, ( double[:], double[:] ), num_threads=16 )    
    
    signature = double[:](double[:],)
    vf = vectorize(['float64(float64)'], nopython=True)(floor_closest_valid_odds)
    
    uf = np.vectorize( floor_closest_valid_odds )
        
    def timefunc(correct, s, func, *args, **kwargs):
        print(s.ljust(20), end=" ")
        # Make sure the function is compiled before we start the benchmark
        res = func(*args, **kwargs)
        if correct is not None:
            assert np.allclose(res, correct), 'results are not all correct'
        # time it
        print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs),
                                              number=5, repeat=2)) * 1000))
        return res
    
    x = np.random.uniform( 1.0, 1000.0, 1e6)
    
    correct = vf( x )
    
    timefunc(correct, "numba (looped)", lf,x)
    timefunc(correct, "numba (vectorised)", vf,x)
    timefunc(correct, "numba (multi-threaded 4 )", mf4,x)
    timefunc(correct, "numba (multi-threaded 6 )", mf6,x)
    timefunc(correct, "numba (multi-threaded 7 )", mf7,x)
    timefunc(correct, "numba (multi-threaded 8 )", mf8,x)
    
    import timeit
    ls = np.logspace(2,7,20)
    
    mf4_results = []
    mf6_results = []
    mf7_results = []
    mf8_results = []
    mf16_results = []
    vf_results = []
    uf_results = []
    lf_results = []
    
    for i, xsize in enumerate( ls ):
        print( 'Generating stats for xsize %s, index %i'%( xsize, i ) )
        x = np.random.uniform( 1.0, 1000.0, int(xsize))
        
        lf_results.append( timeit.timeit( 'lf(x)', "from __main__ import lf,x", number=3) )
        vf_results.append( timeit.timeit( 'vf(x)', "from __main__ import vf,x", number=3) )
       
        mf4_results.append( timeit.timeit( 'mf4(x)', "from __main__ import mf4,x", number=3) )
        mf6_results.append( timeit.timeit( 'mf6(x)', "from __main__ import mf6,x", number=3) )
        mf7_results.append( timeit.timeit( 'mf7(x)', "from __main__ import mf7,x", number=3) )
        mf8_results.append( timeit.timeit( 'mf8(x)', "from __main__ import mf8,x", number=3) )
        mf16_results.append( timeit.timeit( 'mf16(x)', "from __main__ import mf16,x", number=3) )    
        
    import pandas as pd
    print( 'Making Dataframe')
    df = pd.DataFrame( { #'multi-threaded 16':mf16_results,
                        #'multi-threaded 6':mf6_results,
                         #'multi-threaded 7':mf7_results,
                         'multi-threaded 8':mf8_results,
                         'multi-threaded 4':mf4_results,
                         'looped':lf_results,
                         'vectorized':vf_results }, index=ls )
    import matplotlib.pyplot as plt
    print( 'Plotting chart')
    df.plot()
    plt.show( block=False)