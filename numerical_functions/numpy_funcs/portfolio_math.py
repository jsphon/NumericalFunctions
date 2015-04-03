

def portfolio_var( cv, weights ):
    return weights.dot( cv ).dot( weights )

def unweighted_portfolio_var( cv ):
    return cv.sum()