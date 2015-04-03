
from numerical_functions.numpy_funcs.indexing import square_take

def portfolio_var( cv, weights ):
    return weights.dot( cv ).dot( weights )

def unweighted_portfolio_var( cv ):
    return cv.sum()

def unweighted_portfolio_var_by_index( cv, idx ):
    
    return square_take( cv, idx ).sum()