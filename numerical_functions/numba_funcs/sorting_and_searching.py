
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
           
@nb.autojit
def quick_sort(list_):
    """
    Iterative version of quick sort
    """
    
    max_depth = 1000
    
    left        = 0
    right       = list_.shape[0]-1
    
    i_stack_pos = 0
    
    a_temp_stack_left = np.ndarray( max_depth, dtype=np.int32 )
    a_temp_stack_right = np.ndarray( max_depth, dtype=np.int32 )
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