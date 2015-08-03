'''
Created on 1 Aug 2015

@author: jon
'''

import numpy as np

class JaggedArray( object ):
    
    def __init__( self, data, bounds, dtype=np.int ):
        self.data = data
        self.bounds = bounds
        if not isinstance( data, np.ndarray ):
            self.data = np.array( data, dtype=dtype )
        if not isinstance( bounds, np.ndarray ):
            self.bounds = np.array( bounds, dtype=np.int )
            
    def __eq__(self, other):
        #print( 'testing eq')
        if not isinstance( other, JaggedArray ):
            return False
        if len( self )!=len( other ):
            return False
        for i in range( len( self ) ):
            if not np.all( self[i]==other[i] ):
                return False
        return True
        
    def __getitem__( self, i ):
        if isinstance( i, int ):
            i0 = self.bounds[i]
            i1 = self.bounds[i+1]
            return self.data[i0:i1]
        elif isinstance( i, slice ):
            # This returns a view on the array, but with different bounds
            s = slice( i.start, i.stop+1 if i.stop else None, i.step )
            return JaggedArray( self.data, self.bounds[s] )
        elif isinstance( i, tuple ):
            i0 = i[0]
            i1 = i[1]
            if isinstance( i0, slice ) and isinstance( i1, int ):
                # slice on the first index i0, then select the i1-th element
                s0 = self[i0]
                result = np.array( [ row[i1] if row.shape else np.nan for row in s0 ] )
                return result
        else:
            raise NotImplementedError( str( i ) )
    
    def __len__( self ):
        return len( self.bounds ) -1 
    
    def __iter__( self ):
        """ Note that each element of this iterator is a np.ndarray
        """
        for i in range( len( self ) ):
            yield self[i]
      
    def __repr__( self ):
        
        if len(self)>6:
            rows0 = '\n'.join( [ '\t%s,'%x for x in self[:3] ] )
            rows1 = '\n'.join( [ '\t%s,'%x for x in self[-4:] ] )
            return '[\n%s\n\t...\n%s\n]'%(rows0,rows1)
        else:
            rows = '\n'.join( [ '\t%s,'%x for x in self ] )
            return '[\n%s\n]'%rows
    
    def save( self, filename ):
        np.savez( filename, data=self.data, bounds=self.bounds )
        
    @staticmethod
    def load( filename ):
        data = np.load( filename)
        return JaggedArray( data['data'], data['bounds'] )
        
    def save_sep( self, filename ):
        np.save( filename+'.data', self.data )#
        np.save( filename+'.bounds',self.bounds )#{ 'data':self.data, 'bounds':self.bounds } )
        
    def save_compressed( self, f ):
        np.savez_compressed( f, data=self.data, bounds=self.bounds )
    
    @staticmethod
    def from_list( lst, dtype=np.int ):
        bounds    = np.ndarray( len( lst )+1 )
        bounds[0] = 0
        c = 0
        for i, x in enumerate( lst ):
            c+=len(x)
            bounds[i+1]=c
        
        data = [ x for item in lst for x in item ]
        return JaggedArray( data, bounds )

    @property
    def dtype(self):
        return self.data.dtype