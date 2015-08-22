
import numba as nb
import numpy as np

@nb.autojit
def binary_search(alist, item):
    """ An implementation of np.searchsorted
    Returns the left-most element
    Algo from 
    http://rosettacode.org/wiki/Binary_search
    BinarySearch_Left
    """
    if alist.shape[0]==0:
        return -1
    
    first = 0
    last = len(alist)-1

    while first<=last:
        midpoint = (first + last)//2
        if alist[midpoint] >= item:
            last = midpoint-1
        else:
            first = midpoint+1
    return first
           
@nb.autojit( nopython=True )
def quick_sort(list_):
    """
    Iterative version of quick sort
    """
    
    max_depth = 1000
    
    left        = 0
    right       = list_.shape[0]-1
    
    i_stack_pos = 0
    
    a_temp_stack_left = np.empty( max_depth, dtype=np.int32 )
    a_temp_stack_right = np.empty( max_depth, dtype=np.int32 )
    a_temp_stack_left[ i_stack_pos ] = left
    a_temp_stack_right[ i_stack_pos ] = right
    
    i_stack_pos+=1
    #Main loop to pop and push items until stack is empty

    return _quick_sort( list_, a_temp_stack_left, a_temp_stack_right, left, right )

@nb.autojit( nopython=True )
def _quick_sort( list_, a_temp_stack_left, a_temp_stack_right, left, right ):
    
    i_stack_pos = 1
    while i_stack_pos>0:
        
        i_stack_pos-=1
        right = a_temp_stack_right[ i_stack_pos ]
        left  = a_temp_stack_left[ i_stack_pos ]
        
        piv = partition(list_,left,right)
        #If items in the left of the pivot push them to the stack
        if piv-1 > left:            
            a_temp_stack_left[ i_stack_pos ] = left
            a_temp_stack_right[ i_stack_pos ] = piv-1
            i_stack_pos+=1
        if piv+1 < right:
            a_temp_stack_left[ i_stack_pos ] = piv+1
            a_temp_stack_right[ i_stack_pos ] = right
            i_stack_pos+=1
 
@nb.autojit( nopython=True )
def partition(list_, left, right):
    """
    Partition method
    """
    #Pivot first element in the array
    piv = list_[left]
    i = left + 1
    j = right
 
    while 1:
        while i <= j  and list_[i] <= piv:
            i +=1
        while j >= i and list_[j] >= piv:
            j -=1
        if j <= i:
            break
        #Exchange items
        list_[i], list_[j] = list_[j], list_[i]
    #Exchange pivot to the right position
    list_[left], list_[j] = list_[j], list_[left]
    return j


@nb.autojit
def quick_arg_sort(list_):
    """
    Iterative version of quick sort
    """
    
    max_depth = 1000
    
    left        = 0
    right       = list_.shape[0]-1
    
    i_stack_pos = 0
    
    a_temp_stack_left = np.empty( max_depth, dtype=np.int32 )
    a_temp_stack_right = np.empty( max_depth, dtype=np.int32 )
    a_temp_stack_left[ i_stack_pos ] = left
    a_temp_stack_right[ i_stack_pos ] = right
    
    i_stack_pos+=1
    #Main loop to pop and push items until stack is empty

    args = np.arange( list_.shape[0] )
    _quick_arg_sort( list_, args, a_temp_stack_left, a_temp_stack_right, left, right )
    return args


@nb.autojit( nopython=True )
def _quick_arg_sort( list_, args, a_temp_stack_left, a_temp_stack_right, left, right ):
    
    i_stack_pos = 1
    while i_stack_pos>0:
        
        i_stack_pos-=1
        right = a_temp_stack_right[ i_stack_pos ]
        left  = a_temp_stack_left[ i_stack_pos ]
        
        piv = quick_arg_partition(list_,args,left,right)
        #If items in the left of the pivot push them to the stack
        if piv-1 > left:            
            a_temp_stack_left[ i_stack_pos ] = left
            a_temp_stack_right[ i_stack_pos ] = piv-1
            i_stack_pos+=1
        if piv+1 < right:
            a_temp_stack_left[ i_stack_pos ] = piv+1
            a_temp_stack_right[ i_stack_pos ] = right
            i_stack_pos+=1    

@nb.autojit( nopython=True )
def quick_arg_partition( lst, arg, left, right ):
    """
    Partition method
    """
    #Pivot first element in the array
    piv = lst[left]
    i = left + 1
    j = right

    while True:
        while i <= j  and lst[i] <= piv:
            i +=1
        while j >= i and lst[j] >= piv:
            j -=1
        if j <= i:
            break
        #Exchange items
        lst[i], lst[j] = lst[j], lst[i]
        arg[i], arg[j] = arg[j], arg[i]
    #Exchange pivot to the right position
    lst[left], lst[j] = lst[j], lst[left]
    arg[left], arg[j] = arg[j], arg[left]
    return j

@nb.jit( nopython=True )
def merge( x ):
    
    n = x.shape[0]
    width=1
    
    r   = x.copy()
    tgt = np.empty_like( r )
    while width<n:
        i=0
        while i<n:
            istart = i
            imid = i+width
            iend = imid+width
            # i has become i+2*width
            i = iend

            if imid>n:
                imid = n
                
            if iend>n:
                iend=n
            _merge( r, tgt, istart, imid, iend)

        # Swap them round, so that the partially sorted tgt becomes the result,
        # and the result becomes a new target buffer
        r, tgt = tgt, r
        width*=2
        
    return r

@nb.jit( nopython=True )
def _merge( src_arr, tgt_arr, istart, imid, iend ):
    """ The merge part of the merge sort """
    i0   = istart
    i1   = imid
    for ipos in range( istart, iend ):
        if ( i0<imid ) and ( ( i1==iend ) or ( src_arr[ i0 ] < src_arr[ i1 ] ) ):
            tgt_arr[ ipos ] = src_arr[ i0 ]
            i0+=1
        else:
            tgt_arr[ ipos ] = src_arr[ i1 ]
            i1+=1

@nb.jit( nopython=True )
def merge2( x ):
    
    n = x.shape[0]
    width=1
    
    #r   = x.copy()
    r=x
    tgt = np.zeros_like( r )
    while width<n:
        i=0
        while i<n:
            istart = i
            imid = i+width
            iend = imid+width
            # i has become i+2*width
            i = iend

            if imid>n:
                imid = n
                
            if iend>n:
                iend=n
            _merge2( r, tgt, istart, imid, iend)

        # Swap them round, so that the partially sorted tgt becomes the result,
        # and the result becomes a new target buffer
        r, tgt = tgt, r
        width*=2
        
    return r

@nb.jit( nopython=True )
def _merge2( src_arr, tgt_arr, istart, imid, iend ):
    """ The merge part of the merge sort """
    i0   = istart
    i1   = imid
    ipos = i0
    v0 = src_arr[ i0 ]
    v1 = src_arr[ i1 ]
    while i0<imid and i1<iend:        
        if v0<=v1:
            tgt_arr[ ipos ] = v0
            i0+=1
            v0 = src_arr[ i0 ]
        else:
            tgt_arr[ ipos ] = v1
            i1+=1
            v1 = src_arr[ i1 ]
        ipos+=1
        
    while i0<imid:
        tgt_arr[ ipos ] = src_arr[ i0 ]
        ipos+=1
        i0+=1
        
    while i1<iend:
        tgt_arr[ ipos ] = src_arr[ i1 ]
        ipos+=1
        i1+=1

@nb.jit( nopython=True )
def _mergesort_recursive(x, lo, hi, buffer):
    if hi - lo <= 1:
        return
    # Python ints don't overflow, so we could do mid = (hi + lo) // 2
    mid = lo + (hi - lo) // 2
    _mergesort_recursive(x, lo, mid, buffer)
    _mergesort_recursive(x, mid, hi, buffer)
    buffer[:mid-lo] = x[lo:mid]
    read_left = 0
    read_right = mid
    write = lo
    while read_left < mid - lo and read_right < hi:
        if x[read_right] < buffer[read_left]:
            x[write] = x[read_right]
            read_right += 1
        else:
            x[write] = buffer[read_left]
            read_left += 1
        write += 1
    # bulk copy of left over entries from left subarray
    x[write:read_right] = buffer[read_left:mid-lo]
    # Left over entries in the right subarray are already in-place

def mergesort_recursive(x):
    # Copy input array and flatten it
    x = np.array(x, copy=True).ravel()
    n = x.size
    _mergesort_recursive(x, 0, n, np.empty(shape=(n//2,), dtype=x.dtype))
    return x
    