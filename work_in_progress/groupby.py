from numba import autojit
import numpy as np
import pandas as pd

@autojit
def gb_max( grps, fld ):
    a_odds = grps.obj[ fld ].values
    comp_ids, _, ngroups = grps.grouper.group_info
    
    result    = np.ndarray( ngroups, dtype=a_odds.dtype )
    result[:] = np.finfo( np.float ).min
   
    for i in range( len( comp_ids ) ):        
        cid = comp_ids[i]
        if a_odds[i]>result[cid]:
            result[ cid ] = a_odds[i]
    return result
        
@autojit
def gb_min( grps, fld ):
    a_odds = grps.obj[ fld ].values
    comp_ids, _, ngroups = grps.grouper.group_info
    
    result    = np.ndarray( ngroups, dtype=a_odds.dtype )
    result[:] = np.finfo( np.float ).min
   
    for i in range( len( comp_ids ) ):        
        cid = comp_ids[i]
        if a_odds[i]<result[cid]:
            result[ cid ] = a_odds[i]
    return result

@autojit
def gb_sum( grps, fld ):
    a_odds = grps.obj[ fld ].values
    comp_ids, _, ngroups = grps.grouper.group_info
    
    result    = np.ndarray( ngroups, dtype=a_odds.dtype )
    result[:] = 0.0
   
    for i in range( len( comp_ids ) ):        
        cid            = comp_ids[i]
        result[ cid ] += a_odds[i]
    return result

@autojit
def arr_gb_sum( comp_ids, ngroups, data ):
    """ Groupby Sum, but taking taking arrays as input """        
    
    result    = np.ndarray( ngroups, dtype=data.dtype )
    result[:] = 0.0
   
    for i in range( len( comp_ids ) ):        
        cid            = comp_ids[i]
        result[ cid ] += data[i]
    return result

@autojit
def gb_count( grps ):    
    comp_ids, _, ngroups = grps.grouper.group_info    
    result    = np.zeros( ngroups, dtype=np.int )   
   
    for i in range( len( comp_ids ) ):        
        cid = comp_ids[i]
        result[ cid ]+=1
        
    return result


def gb_count_filtered_as_frame( grps, flt ):
    r = gb_count_filtered( grps, flt )
    return pd.DataFrame( {'count':r}, grps.grouper.result_index )

@autojit
def gb_count_filtered( grps, flt ):    
    comp_ids, _, ngroups = grps.grouper.group_info    
    result    = np.zeros( ngroups, dtype=np.int )   
   
    for i in range( len( comp_ids ) ):
        if flt[i]:        
            cid = comp_ids[i]
            result[ cid ]+=1
        
    return result


def find_min_max_filtered_as_frame( grps, fld, flt ):
    r_argmin, r_min, r_argmax, r_max = find_min_max_filtered( grps, fld, flt )
    return pd.DataFrame( { 'argmin':r_argmin, 'min':r_min, 'argmax':r_argmax, 'max':r_max }, grps.grouper.result_index )
    

@autojit
def find_min_max_filtered( grps, fld, flt ):
    """r_argmin, r_min, r_argmax, r_max = find_min_max_filtered( grps, fld, flt )
    Find the min/max of each group
    """
    
    comp_ids, _, ngroups = grps.grouper.group_info
    
    x = grps.obj[ fld ].values
    
    dtype = grps.obj[ fld ].dtype
    r_min = np.ndarray( ngroups, dtype=dtype ) 
    r_max = np.ndarray( ngroups, dtype=dtype )
    
    if np.issubdtype( dtype, np.float ):
        isFloatingType=True
        r_min[:] = np.finfo( dtype ).max
        r_max[:] = np.finfo( dtype ).min
    elif np.issubdtype( dtype, np.int ):    
        isFloatingType=False
        r_min[:] = np.iinfo( dtype ).max
        r_max[:] = np.iinfo( dtype ).min
    #else:
    #    raise Exception( 'dtype %s not recognised'%dtype )
            
    r_argmin = np.ndarray( ngroups, dtype=np.int )
    r_argmax = np.ndarray( ngroups, dtype=np.int )
    
    r_argmin[:]=np.iinfo( np.int ).max
    r_argmax[:]=np.iinfo( np.int ).min
    
    for i in range( len( comp_ids ) ):
        if flt[i]:
            cid  = comp_ids[ i ]
            o    = x[ i ]
        
            if o<r_min[cid]:
                r_argmin[cid] = i
                r_min[cid] = o
                
            if o>r_max[cid]:
                r_argmax[cid]=i
                r_max[cid]=o
            
    if isFloatingType:
        r_min[ np.isnan( r_argmin ) ] = np.nan
        r_max[ np.isnan( r_argmax ) ] = np.nan
        
            
    return r_argmin, r_min, r_argmax, r_max

def find_min_max_as_frame( grps, fld  ):
    r_argmin, r_min, r_argmax, r_max = find_min_max( grps, fld )
    return pd.DataFrame( { 'argmin':r_argmin, 'min':r_min, 'argmax':r_argmax, 'max':r_max }, grps.grouper.result_index )
 
@autojit
def find_min_max( grps, fld  ):
    """r_argmin, r_min, r_argmax, r_max = find_min_max( grps, fld )
    Find the min/max of each group
    """
    
    comp_ids, _, ngroups = grps.grouper.group_info
    
    x = grps.obj[ fld ].values
    
    r_min = np.ndarray( ngroups, dtype=np.float ) 
    r_max = np.ndarray( ngroups, dtype=np.float )
    
    r_min[:] = np.finfo( np.float ).max
    r_max[:] = np.finfo( np.float ).min    
    
    r_argmin = np.ndarray( ngroups, dtype=np.int )
    r_argmax = np.ndarray( ngroups, dtype=np.int )
    
    r_argmin[:]=np.iinfo( np.int ).max
    r_argmax[:]=np.iinfo( np.int ).min
    
    for i in range( len( comp_ids ) ):       
        cid  = comp_ids[ i ]
        o    = x[ i ]
    
        if o<r_min[cid]:
            r_argmin[cid] = i
            r_min[cid] = o
            
        if o>r_max[cid]:
            r_argmax[cid]=i
            r_max[cid]=o
            
    r_min[ np.isnan( r_argmin ) ] = np.nan
    r_max[ np.isnan( r_argmax ) ] = np.nan
            
    return r_argmin, r_min, r_argmax, r_max