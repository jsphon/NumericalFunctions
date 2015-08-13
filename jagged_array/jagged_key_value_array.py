'''
Created on 1 Aug 2015

@author: jon
'''

from jagged_array.jagged_array import JaggedArray
import numba as nb
import numpy as np

INT_TYPES = ( int, np.int, np.int64 )

class JaggedKeyValueArray( object ):
    
    def __init__( self, keys, values, bounds, dtype=np.int ):
        
        if isinstance( keys, np.ndarray ):
            self.keys = keys
        else:
            self.keys = np.array( keys, dtype=dtype )
            
        if isinstance( values, np.ndarray ):
            self.values = values
        else:
            self.values = np.array( values, dtype=dtype )
            
        if isinstance( bounds, np.ndarray ):
            self.bounds = bounds
        else:
            self.bounds = np.array( bounds, dtype=np.int )
            
    def __len__(self):
        return self.bounds.shape[0]-1
            
    def __eq__(self, other):
        #print( 'testing eq')
        if not isinstance( other, JaggedKeyValueArray ):
            return False
        if len( self )!=len( other ):
            return False
        for i in range( len( self ) ):
            if not np.all( self[i]==other[i] ):
                return False
        return True
    
    def __getitem__( self, i ):
        
        if isinstance( i, INT_TYPES ):
            i0 = self.bounds[i]
            i1 = self.bounds[i+1]
            return ( self.keys[i0:i1], self.values[i0:i1] )
        
        if isinstance( i, slice ):
            s = slice( i.start, i.stop+1 if i.stop else None, i.step )
            return JaggedKeyValueArray( self.keys, self.values, self.bounds[s] )
        
        if isinstance( i, tuple ):
            #print( 'i is a tuple')
            i0 = i[0]
            i1 = i[1]
            if isinstance( i0, INT_TYPES ):
                if isinstance( i1, INT_TYPES ) or isinstance( i1, slice ):
                    j0 = self.bounds[i0]
                    j1 = self.bounds[i0+1]
                    return ( self.keys[j0:j1][i1], self.values[j0:j1][i1] )
        
        raise Exception( 'Not implemented for slice %s'%str(i))
    def __repr__( self ):
        
        if len(self)>6:
            rows0 = '\n'.join( [ '\t%s,'%x for x in self[:3] ] )
            rows1 = '\n'.join( [ '\t%s,'%x for x in self[-4:] ] )
            return '[\n%s\n\t...\n%s\n]'%(rows0,rows1)
        else:
            rows = '\n'.join( [ '\t%s,'%str(x) for x in self ] )
            return '[\n%s\n]'%rows
    
    def get_keys_array(self):
        """ Return a jagged array of the keys """
        return JaggedArray( self.keys, self.bounds )
    
    def get_values_array(self):
        """ Return a jagged array of values """
        return JaggedArray( self.values, self.bounds )
    
        
    
    
def kv_to_dense( key_data, val_data, bounds, default_value=0 ):

    unique_keys, inverse_data = np.unique( key_data, return_inverse=True)
    
    data = _kv_to_dense( key_data, val_data, bounds, unique_keys, inverse_data, default_value )
    
    return data, unique_keys

@nb.jit( nopython=True )
def _kv_to_dense( key_data, val_data, bounds, unique_keys, inverse_data, default_value ):
    
    data = np.zeros( ( len( bounds )-1, len( unique_keys ) ), dtype=key_data.dtype)
    for i in range( bounds.shape[0]-1 ):
        i0 = bounds[i]
        i1 = bounds[i+1]
        
        for j in range( i1-i0 ):
            col_idx            = inverse_data[ i0+j ]
            data[ i, col_idx ] = val_data[ i0 + j ]
    
    return data 

def dense_to_kv( data, cols ):
    
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