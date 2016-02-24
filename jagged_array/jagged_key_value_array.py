'''
Created on 1 Aug 2015

@author: jon
'''

from jagged_array.jagged_array import JaggedArray
import numba as nb
import numpy as np
from numerical_functions import numba_funcs as nf

INT_TYPES = ( int, np.int, np.int64 )

class JaggedKeyValueArray( object ):
    
    @nb.jit
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
            
    @staticmethod
    def from_lists( key_list, val_list, dtype=np.int ):
        """ Make a JaggedKeyValueArray from key and value list of lists
        """
        assert len( key_list )==len( val_list )
        
        bounds    = np.ndarray( len( key_list )+1, dtype=np.int )
        bounds[0] = 0
        c = 0
        for i, x in enumerate( key_list ):
            c+=len(x)
            bounds[i+1]=c
        
        key_list = [ x for item in key_list for x in item ]
        val_list = [ x for item in val_list for x in item ]
        return JaggedKeyValueArray( key_list, val_list, bounds )
    
    @staticmethod
    @nb.jit
    def from_dense_nb( data, cols ):#, dtype=np.int ):
        """ Make a JaggedKeyValueArray from a dense array """    
        keys, values, bounds=_from_dense_nb( data, cols )#, dtype)
        n = bounds[-1]
        return JaggedKeyValueArray( keys[:n], values[:n], bounds )

    @staticmethod
    def from_dense( data, cols, dtype=np.int ):
        """ Make a JaggedKeyValueArray from a dense array """
    
        keys = []
        values = []
        bounds = []
        for i in range( len( data ) ):
            row = data[ i ]
            mask = ( row!=0 )
            row_vals = row.compress( mask )
            row_cols = cols.compress( mask )
            
            keys.append( row_cols )
            values.append( row_vals )
            bounds.append( len( row_cols ) )
        
        return JaggedKeyValueArray.from_lists( keys, values )
    
    def __bool__(self):
        return bool( self.keys.shape[0] )
    
    def __len__(self):
        return self.bounds.shape[0]-1
            
    def __eq__(self, other):
        #print( 'testing eq')
        if not isinstance( other, JaggedKeyValueArray ):
            return False
        if len( self )!=len( other ):
            return False
        
        if not np.array_equal( self.bounds, other.bounds ):
            return False
        if not np.array_equal( self.keys, other.keys ):
            return False
        if not np.array_equal( self.values, other.values ):
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
        
            if isinstance( i0, slice ):
                row_start = i0.start or 0
                row_end   = i0.stop or len( self )
                if isinstance( i1, slice ):
                    # Return fixed size arrays for the keys and values
                    assert i1.step is None, 'Only step 1 supported, step is %s'%i1.step
                    assert (i1.start is None) or (i1.stop is None) 
                    
                    if i1.start:
                        start = i1.start
                        assert start<0
                        size = -start
                        key_result = np.zeros( ( len( self ), size ) )
                        val_result = np.zeros( ( len( self ), size ) )
                        
                        ii=0
                        for row in range( row_start, row_end+1 ):
                            row_key, row_val = self[row]
                            rk = row_key[start:]
                            key_result[ii, start:]=rk
                            if rk.shape[0]<size:
                                key_result[ii,:size-rk.shape[0]]=0
                            rv = row_val[start:]
                            val_result[ii, start:]=rv
                            if rv.shape[0]<size:
                                val_result[ii,:size-rv.shape[0]]=0
                                
                            ii+=1
                        
                        return key_result, val_result
                    
                elif isinstance( i1, INT_TYPES ):
                    key_result = np.zeros( len( self ) )
                    val_result = np.zeros( len( self ) )
                    
                    for i, row in enumerate( range( row_start, row_end ) ):
                        row_key, row_val = self[row]
                        key_result[i]=row_key[i1]
                        val_result[i]=row_val[i1]
                    return key_result, val_result        
                           
        raise Exception( 'Not implemented for slice %s'%str(i))
    def __repr__( self ):
        
        if len(self)>6:
            rows0 = '\n'.join( [ '\t%s,%s,'%x for x in self[:3] ] )
            rows1 = '\n'.join( [ '\t%s,%s,'%x for x in self[-4:] ] )
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
    
    def to_dense_projection( self, projection, default_value=0 ):
        ''' Convert to a dense array, using projection as the keys
        '''
        d, k = self.to_dense( default_value=default_value )
        return _kv_to_dense_projection( d, k, projection )

    def to_dense( self, default_value=0 ):
    
        i0 = self.bounds[0]
        i1 = self.bounds[-1]
        keys = self.keys[i0:i1]
        vals = self.values[i0:i1]
        bounds = self.bounds - i0
        unique_keys, inverse_data = np.unique( keys, return_inverse=True)
        
        data = _kv_to_dense( keys, vals, bounds, unique_keys, inverse_data, default_value )
        
        return data, unique_keys
    
    def cumsum(self):
        unique_keys, inverse_data = np.unique( self.keys, return_inverse=True)
        
        if self:
            cs_keys, cs_vals, cs_bounds = _cumsum(self.keys, self.values, self.bounds, unique_keys, inverse_data)
            return JaggedKeyValueArray( cs_keys, cs_vals, cs_bounds )
        else:
            return JaggedKeyValueArray( [], [], [] )
        
@nb.jit( nopython=True )
def _cumsum( keys, values, bounds, unique_keys, inverse_data ):
    buffer = np.zeros_like( unique_keys, np.int_ )
    max_possible_length = ( bounds.shape[0]-1 ) * unique_keys.shape[0]
    cs_keys = np.empty( max_possible_length, dtype=keys.dtype )
    cs_vals = np.empty( max_possible_length, dtype=values.dtype )
    cs_bounds = np.empty_like( bounds )
    cs_bounds[0] = 0
    pos = 0
    for i in range( bounds.shape[0]-1 ):
        i0 = bounds[i]
        i1 = bounds[i+1]
        for j in range( i0, i1 ):
            bcol = inverse_data[j]
            buffer[ bcol ] += values[ j ]
            
        for j in range( unique_keys.shape[0] ):
            if buffer[ j ]:
                cs_vals[ pos ] = buffer[ j ]
                cs_keys[ pos ] = unique_keys[ j ]
                pos+=1
        cs_bounds[i+1]=pos
            
    return cs_keys[:pos].copy(), cs_vals[:pos].copy(), cs_bounds

@nb.jit( nopython=True )
def _kv_to_dense_projection( d, k, projection ):
    r = np.zeros( ( d.shape[0], projection.shape[0] ), d.dtype )
    for i in range( r.shape[1] ):
        v = projection[ i ]
        j    = nf.binary_search( k, v )
        if (j<k.shape[0]) and ( k[ j ]==v ):
            for row in range( d.shape[0] ):
                r[ row, i ] = d[ row, j ]
    return r
    
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

@nb.jit( nopython=True )
def _from_dense_nb( data, cols):#, dtype=np.int ):
    """ Make the params for from_dense_nb """
    l = data.size
    keys   = np.empty( l, cols.dtype )
    values = np.empty( l, data.dtype )
    bounds = np.empty( data.shape[0]+1, np.int_ )
    bounds[0] = 0
    c = 0
    for i in range( data.shape[0] ):
        row = data[ i ]
        for j in range( data.shape[1] ):
            if row[ j ]:
                keys[ c ] = cols[ j ]
                values[ c ] = row[ j ]
                c+=1
        bounds[ i+1 ]=c
    return keys, values, bounds