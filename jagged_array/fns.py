'''
Created on 1 Aug 2015

@author: jon
'''

import numpy as np
import pandas as pd
from jagged_array.jagged_array import JaggedArray

def to_frame( keys, values, default_value=0 ):

    unique_keys, inverses = np.unique( keys.data, return_inverse=True)
    
    inverse_array = JaggedArray( inverses, keys.bounds )
    
    data = np.ndarray( ( len( keys ), len(unique_keys ) ), dtype=values.dtype)
    data[:]=default_value
    for i in range( len( keys ) ):
        row  = data[i]
        keys = inverse_array[i]
        vals = values[i]
        row[keys]=vals
    
    result = pd.DataFrame( data,columns=unique_keys)
    
    return result

def frame_to_jagged_array( df ):
    
    cols = df.columns.values
    data = df.values
    
    keys = []
    values = []
    for i in range( len( data ) ):
        row = data[ i ]
        mask = ( row!=0 )
        #print(mask)
        row_vals = row.compress( mask )
        row_cols = cols.compress( mask )
        
        keys.append( row_cols )
        values.append( row_vals )
        
    ka = JaggedArray.from_list( keys )
    va = JaggedArray.from_list( values )
    
    return ka, va

def accumulate( keys, values ):
    key_result = []
    val_result = []
    for i in range( len( keys ) ):
        #print(i,len(price_result))
        pr = keys[i]
        sr = values[i]
        if i==0:       
            key_result.append( list( pr ) )
            val_result.append( list( sr ) )

            diffs = dict(zip(pr,sr))
        else:        
            for p, s in zip( *[ pr, sr ] ):
                diffs[ p ] = diffs.get( p, 0 ) + s
            row_keys = sorted(diffs.keys(),reverse=True)

            r = [ ( k, diffs[k] ) for k in row_keys ]
            r = [ ( k, d ) for k, d in r if d ]
            
            key_result.append( [ k for k, _ in r ] )
            val_result.append( [ d for _, d in r ] )

    ka = JaggedArray.from_list( key_result )
    va = JaggedArray.from_list( val_result )
    return ka, va