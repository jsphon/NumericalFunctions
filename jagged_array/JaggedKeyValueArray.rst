
JaggedKeyValueAray
==================

This is a variation of a sparse matrix or Pandas dataframe.

Unlike sparse matrices, it has labelled columns.

Unlike sparse dataframes, it provides an interface for efficient
iteration, over rows, of the non-zero values.

.. code:: python

    from jagged_array.jagged_key_value_array import JaggedKeyValueArray
    
    
    keys   = [ [ 1 ],  [ 1 , 2 , 3 ],  [ 2 , 3 ] ]
    values = [ [ 10 ], [ 21, 22, 23 ], [ 32, 33 ] ]
    
    arr = JaggedKeyValueArray.from_lists( keys, values )
    arr




.. parsed-literal::

    [
    	(array([1]), array([10])),
    	(array([1, 2, 3]), array([21, 22, 23])),
    	(array([2, 3]), array([32, 33])),
    ]



An example of array access:

.. code:: python

    print( 'arr[0]: %s'%str(arr[0]))


.. parsed-literal::

    arr[0]: (array([1]), array([10]))


An example of array slicing:

.. code:: python

    arr[:2]




.. parsed-literal::

    [
    	(array([1]), array([10])),
    	(array([1, 2, 3]), array([21, 22, 23])),
    ]



An example of iterating over the array:

.. code:: python

    for row_keys, row_vals in arr:
        print( 'keys: %s, values: %s'%( row_keys, row_vals  ) )


.. parsed-literal::

    keys: [1], values: [10]
    keys: [1 2 3], values: [21 22 23]
    keys: [2 3], values: [32 33]


.. code:: python

    data, cols = arr.to_dense()
    print( data )
    print( cols )


.. parsed-literal::

    [[10  0  0]
     [21 22 23]
     [ 0 32 33]]
    [1 2 3]


.. code:: python

    arr2 = JaggedKeyValueArray.from_dense( data, cols )
    assert arr==arr2
    arr2




.. parsed-literal::

    [
    	(array([1]), array([10])),
    	(array([1, 2, 3]), array([21, 22, 23])),
    	(array([2, 3]), array([32, 33])),
    ]



.. code:: python

    from numerical_functions.misc.timer import Timer
    
    with Timer( 'cumsum' ):
        data, cols = arr.to_dense()
        dense_cs = data.cumsum( axis=0 )
        sparse_cs = JaggedKeyValueArray.from_dense( dense_cs, cols )
    print( sparse_cs )


.. parsed-literal::

    Beginning cumsum
    cumsum took 0.0005 seconds
    [
    	(array([1]), array([10])),
    	(array([1, 2, 3]), array([31, 22, 23])),
    	(array([1, 2, 3]), array([31, 54, 56])),
    ]

