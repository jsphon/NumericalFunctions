

def square_take( x, idx ):
    return x.take( idx, axis=0 ).take( idx, axis=1 )