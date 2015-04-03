import numba as nb

@nb.jit( nopython=True )
def add( a, b ):
    return a+b