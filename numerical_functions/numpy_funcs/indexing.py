

def square_take( x, idx ):
    return x.take( idx, axis=0 ).take( idx, axis=1 )

def swap_row_cols( X, i, j ):
    """ Swap the rows and cols of X indexed by i and j """
    t      = X[ :,j ].copy()
    X[:,j] = X[:,i]
    X[:,i] = t
    
    t = X[ j, : ].copy()
    X[j,:] = X[i,:]
    X[i,:] = t