 
import numpy as np
import pandas as pd
from numba import void, double, jit,int64, autojit

import threading
from ctypes import pythonapi, c_void_p
from timer import Timer

import multiprocessing

def make_multithreaded_groupby( fn, agg_func, default_value, data_type=None, numthreads=multiprocessing.cpu_count() ):
    if not data_type:
        assert len( fn.signatures )==1, 'You should not provide a function with multiple signatures if you don''t provide the data type'
        data_type = fn.signatures[0][0].dtype

    inner_func = make_inner_func( fn, data_type )
    fn = make_multithread( inner_func, agg_func, numthreads, default_value )
    return fn

def make_singlethreaded_groupby( fn, default_value, data_type=None ):
    if not data_type:
        assert len( fn.signatures )==1, 'You should not provide a function with multiple signatures if you don''t provide the data type'
        data_type = fn.signatures[0][0].dtype

    inner_func = make_inner_func( fn, data_type )
    
    def func_st( grps, fld ):
        
        comp_ids, _, ngroups = grps.grouper.group_info
        
        result = np.empty(ngroups, dtype=np.float64)
        result[:] = default_value
        
        data = grps.obj[ fld ].values
        
        inner_func( result, comp_ids, data )
        
        return result
        
    return func_st
    
def make_inner_func( fn, data_type ):
    signature = void( data_type[:], int64[:], data_type[:] )
    @jit(signature)
    def inner_func( result, comp_ids, data ):
        threadstate = savethread()
        fn( result, comp_ids, data )
        restorethread(threadstate)
    return inner_func
    
def make_multithread(inner_func, agg_func, numthreads, default_value):
    def func_mt( grps, fld ):
        
        comp_ids, _, ngroups = grps.grouper.group_info
        
        result = np.empty( (ngroups, numthreads ), dtype=np.float64)
        result[:] = default_value

        data = grps.obj[ fld ].values
        n_length_args = (comp_ids,) + ( data, )
        idx    = np.arange( len( data ) )
        splits = np.array_split( idx, numthreads)
        chunks = [ [result[:,i]] + [arg[s[0]:s[-1]+1] for arg in n_length_args]
                  for i, s in enumerate( splits ) ]
        
        #print( chunks )
    
        # You should make sure inner_func is compiled at this point, because
        # the compilation must happen on the main thread. This is the case
        # in this example because we use jit().
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks[:-1]]
        for thread in threads:
            thread.start()

        # the main thread handles the last chunk
        inner_func(*chunks[-1])

        for thread in threads:
            thread.join()
        return agg_func( result )
    
    return func_mt

def get_group_splits( grps, num_splits ):
    comp_ids, _, _ = grps.grouper.group_info
    length               = len( comp_ids )
    
    arrs              = np.array_split( comp_ids, num_splits )
    borders_comp_ids  = [a[0] for a in arrs[1:]]
    borders_indices   = np.searchsorted( comp_ids, borders_comp_ids )
    borders_indices   = np.append( np.insert( borders_indices, 0, 0), length )
    
    return borders_indices

@jit(void(double[:], int64[:], double[:]),nopython=True)
def pgb_max( result, comp_ids, data ):
    for i in range( len( comp_ids ) ):        
        cid = comp_ids[i]
        ai  = data[i]
        if ai>result[cid]:
            result[ cid ] = ai
            
@jit(void(double[:], int64[:], double[:]),nopython=True)
def pgb_sum( result, comp_ids, data ):
    for i in range( len( comp_ids ) ):        
        cid = comp_ids[i]
        result[ cid ] += data[i]
        
@jit(double[:](double[:,:]))
def row_sum( d ):
    result = np.zeros( d.shape[0] )
    cols = d.shape[1]
    for i in range( d.shape[0] ):
        for j in range( cols ):
            result[i]+=d[i,j]
    return result

@jit(double[:](double[:,:]))
def row_max( d ):
    result = np.zeros( d.shape[0] )
    cols = d.shape[1]
    mx0 = np.finfo( np.float ).min
    for i in range( d.shape[0] ):
        mx = mx0
        for j in range( cols ):
            if d[i,j]>mx:
                mx = d[i,j]
        result[ i ] = mx
    return result
    
savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p

restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None

if __name__=='__main__':
    
    N = 1e5
    m = N/10
    p = 3
    
    x = np.random.randint( 0, m, N )
    y = np.random.randint( 0, p, N )
    z = np.random.randn(N)
    
    key = [ 'x', 'y' ]
    fld      = 'z'
    
    df = pd.DataFrame( {'x':x, 'y':y, 'z':z} )
    
    grps = df.groupby( key )
    #print( df.head() )    
    
    with Timer( 'Caching Group Info so that we can do fair comparisons' ):
        comp_ids, _, ngroups = grps.grouper.group_info
    
    agg_func = lambda x: np.max( x, axis=1)
    mfn = make_multithreaded_groupby( pgb_max, row_max, np.finfo( np.float ).min )
    sfn = make_singlethreaded_groupby( pgb_max, np.finfo( np.float ).min )
    
    agg_func_sum = lambda x: np.sum( x, axis=1 )
    mfn_sum =  make_multithreaded_groupby( pgb_sum, row_sum, 0.0 )
    sfn_sum = make_singlethreaded_groupby( pgb_sum, 0.0 )
    
    with Timer( 'MT max' ):
        mt_result = mfn( grps, fld )
        
    with Timer( 'MT sum' ):
        mt_sum_result = mfn_sum( grps, fld )
    
    with Timer( 'ST' ):
        st_result = sfn( grps, fld )
        
    with Timer( 'ST sum' ):
        st_sum_result = sfn_sum( grps, fld )
        
    with Timer( 'Pandas' ):
        pd_max = grps[ 'z' ].max().values
        
    with Timer( 'Pandas Sum' ):
        pd_sum = grps[ 'z' ].sum().values
    
    assert np.all(pd_max==mt_result)
    assert np.all(pd_max==st_result)
    
    #np.testing.assert_allclose( pd_sum, mt_sum_result )
    #np.testing.assert_allclose( pd_sum, st_sum_result )
    
    print( 'Values match')
    
    import timeit
    ls = np.logspace(1,8,10)
    
    mt_results = []
    st_results = []
    pd_results = []
    
    for i, xsize in enumerate( ls ):
        print( 'Generating stats for xsize %s, index %i'%( xsize, i ) )
    
        m = xsize/10
        p = 3
        
        x = np.random.randint( 0, m, xsize )
        y = np.random.randint( 0, p, xsize )
        z = np.random.randn(xsize)
        
        df = pd.DataFrame( {'x':x, 'y':y, 'z':z} )    
        grps = df.groupby( key )
        
        with Timer( 'Caching Group Info so that we can do fair comparisons' ):
            comp_ids, _, ngroups = grps.grouper.group_info
        
        mt_results.append( timeit.timeit( 'mfn(grps, fld)', "from __main__ import mfn,grps, fld", number=3) )
        st_results.append( timeit.timeit( 'sfn(grps, fld)', "from __main__ import sfn, grps, fld", number=3) )        
        
        pd_results.append( timeit.timeit( 'grps[ fld ].max().values', "from __main__ import grps, fld", number=3) )
        
    df = pd.DataFrame( {'multi-threaded':mt_results, 'single-threaded':st_results, 'pandas':pd_results }, index=ls )
    import matplotlib.pyplot as plt
    df.plot()
    plt.show()
    

