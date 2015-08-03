
Jagged Array
============

An implementation of a jagged array.

.. code:: python

    from jagged_array.jagged_array import JaggedArray
    
    x0 = [ 1, ]
    x1 = [ 2, 3 ]
    x2 = [ 4, 5, 6 ]
    x3 = [1,2,3,4,5,6,7,8,9,10]
    
    data   = x0+x1+x2+x3
    bounds = [ 0, 1, 3, 6,16 ]
    
    ja = JaggedArray( data, bounds )
    print( ja )


.. parsed-literal::

    [
    	[1],
    	[2 3],
    	[4 5 6],
    	[ 1  2  3  4  5  6  7  8  9 10],
    ]


Iteration
=========

You can iterate over a jagged array, to return the rows. Each row will
be a numpy array.

.. code:: python

    for row in ja:
        print( row )


.. parsed-literal::

    [1]
    [2 3]
    [4 5 6]
    [ 1  2  3  4  5  6  7  8  9 10]


Slicing
=======

You can access rows through indices or slicing.

Using an integer index will return a numpy array:

.. code:: python

    print( ja[0] )


.. parsed-literal::

    [1]


Using a slice will return another Jagged Array, which will be a view on
the original array.

.. code:: python

    print( ja[:2])


.. parsed-literal::

    [
    	[1],
    	[2 3],
    ]


You can also slice along the second axis.

.. code:: python

    print( ja[:,0] )


.. parsed-literal::

    [1 2 4 1]


.. code:: python

    print( ja[:,-1] )


.. parsed-literal::

    [ 1  3  6 10]

