'''
Created on 1 May 2015

@author: jon
'''

from datetime import timedelta
import numpy as np
import pandas as pd

def bdate_range_slicer( dmin=pd.datetime(1980,1,1), dmax=pd.datetime.today()+timedelta(days=1) ):
    """ Return a bdate_range function
    that generates the range using a slice from a
    pre-defined range
    This should be a lot faster than pd.bdate_range
    """
    dr = pd.bdate_range( dmin, dmax )
    
    def bdate_range( fd, ld ):
        fd64 = np.datetime64(fd)
        ld64 = np.datetime64(ld)
    
        fd_idx = np.searchsorted( dr.values, fd64 )
        ld_idx = np.searchsorted( dr.values, ld64 )
        
        if dr[ ld_idx ]==ld64:
            ld_idx+=1
    
        return dr[fd_idx:ld_idx]
    
    return bdate_range

bdate_range = bdate_range_slicer()
