# Reproducing some numpy functions...
# https://github.com/numpy/numpy/blob/master/numpy/lib/arraysetops.py

import numerical_functions.numba_funcs.sorting_and_searching as sas
import numpy as np
import numba as nb
from numba_funcs.sorting_and_searching import quick_sort

@nb.jit( nopython=True )
def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    """
    Find the unique elements of an array.
    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements: the indices of the input array
    that give the unique values, the indices of the unique array that
    reconstruct the input array, and the number of times each unique value
    comes up in the input array.
    Parameters
    ----------
    ar : array_like
        Input array. This will be flattened if it is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` that result in the unique
        array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array that can be used
        to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique value comes up
        in `ar`.
        .. versionadded:: 1.9.0
    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        (flattened) original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the (flattened) original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.
        .. versionadded:: 1.9.0
    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.
    Examples
    --------
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])
    Return the indices of the original array that give the unique values:
    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'],
           dtype='|S1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'],
           dtype='|S1')
    Reconstruct the input array from the unique values:
    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])
    """
    
    #ar = np.asanyarray(ar).flatten()
    l = ar.shape[0]
    for i in range( 1, len( ar.shape ) ):
        l *= ar.shape[i]
    ar = ar.reshape( ( l, ) ).copy()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool_),)
            if return_inverse:
                ret += (np.empty(0, np.bool_),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret

    if optional_indices:
        #perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        #aux = ar[perm]
        perm = sas.quick_arg_sort( ar )
        aux = ar# quick_arg_sort also sorts ar in place, though that is bad form
    else:
        #ar.sort()
        sas.quick_sort( ar )
        aux = ar
        
    #flag = np.concatenate(([True], aux[1:] != aux[:-1]))
    flag = np.empty( ar.shape[0], dtype=np.bool_ )
    for i in range( 1, ar.shape[0] ):
        #if aux[i]!=aux[i-1]:
        flag[i]=( aux[i]!=aux[i-1] )
            
    if not optional_returns:
        #print( 'no optional returns' )
        #print( 'flag is %s'%str(flag))
        ret = aux[flag]
    else:
        #print( 'Making ret' )
        ret = (aux[flag],)
        #print( 'ret is %s'%str(ret))
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            uniques = ret[0]
            cnts    = np.zeros_like( uniques )
            c = 0
            j = 0
            for i in range( uniques.shape[0] ):
                c+=1
                if flag[ i ]:
                    cnts[j]=c
                    c=0
                    
            #idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            #ret += (np.diff(idx),)
            ret += ( cnts )
            pass
    
    return ret

#@nb.jit(nopython=True)
def unique2(ar , return_index=False, return_inverse=False, return_counts=False):
    
    l = ar.shape[0]
    for i in range( 1, len( ar.shape ) ):
        l *= ar.shape[i]
    ar = ar.reshape( ( l, ) ).copy()
    
    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts
    
    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret

    quick_sort( ar )
    
    if ar.shape:
        num_uniques = 1
        for i in range( 1, ar.shape[0] ):
            if ar[i]!=ar[i-1]:
                num_uniques+=1
                
    rr_arr = np.empty( num_uniques, dtype=ar.dtype )
    rr_arr[0] = ar[0]
    c = 1
    for i in range( 1, ar.shape[0] ):
        if ar[i]!=ar[i-1]:
            rr_arr[c]=ar[i]
            c+=1
        
    
    return rr_arr
    