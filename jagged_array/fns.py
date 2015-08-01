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