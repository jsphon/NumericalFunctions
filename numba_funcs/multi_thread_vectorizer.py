# -*- coding: utf-8 -*-
"""
Multi-Threading Vectorizer using Numba
This is basically a modification of the code found here:
http://numba.pydata.org/numba-doc/dev/examples.html
"""


from numba import vectorize
from timeit import repeat
import threading
from ctypes import pythonapi, c_void_p

import numpy as np
from numba import jit, void, double, autojit

import multiprocessing

def mvectorize( fn, signature, num_threads=None, return_type=None ):
    """
    A Multi-Threaded vectorisation decorator
    fn needs to be compiled in Numba before use
    *args consist the return type, followed by the input arguments
    e.g. to vectorise a function that will return a double array, and takes a double array as input,
        mvectorise( fn, double[:], double[:] )
    """
    new_inner_func = make_inner_func( fn, *signature )
    num_threads = num_threads or multiprocessing.cpu_count()
    return make_multithread( new_inner_func, num_threads, return_type )

def make_inner_func( fn, *args ):
    signature = void( *args )
    @jit(signature)
    def inner_func( result, x ):
        threadstate = savethread()
        for i in range(len(result)):
            result[i] = fn( x[i] )
        restorethread(threadstate)
    return inner_func

def make_multithread(inner_func, numthreads, return_type=np.float64 ):
    def func_mt(*args):
        length = len(args[0])
        #result = np.empty(length, dtype=np.float64)        
        result = np.empty(length, dtype=return_type )
        args = (result,) + args        
        #chunklen = (length + 1) // numthreads
        #chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in args]
        #          for i in range(numthreads)]
        
        # Rewrite the above, as it does not work when numthreads does not divide
        # length evenly
        idx    = np.arange( length )
        splits = np.array_split( idx, numthreads)
        chunks = [[arg[s[0]:s[-1]+1] for arg in args]
                  for s in splits ]

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
            
        return result
    return func_mt
  
savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p

restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None