import tempfile
import unittest

import numpy as np

from jagged_array.jagged_key_value_array import JaggedKeyValueArray


NAN = np.nan


class MoreJaggedKeyValueArrayTests(unittest.TestCase):
    def test___repr__(self):
        k0 = [10, 11]
        k1 = [11, 12, 13]
        k2 = [12, 13]
        k3 = []
        k4 = [14]
        k5 = [10, 11]
        k6 = [11, 12, 13]

        v0 = [1, 2]
        v1 = [3, 4, 5]
        v2 = [6, 7]
        v3 = []
        v4 = [8]
        v5 = [1, 2]
        v6 = [3, 4, 5]

        keys = k0 + k1 + k2 + k3 + k4 + k5 + k6
        vals = v0 + v1 + v2 + v3 + v4 + v5 + v6
        bounds = [0, 2, 5, 7, 7, 8, 10, 13]
        arr = JaggedKeyValueArray(keys, vals, bounds)

        result = str(arr)
        print(result)

    def test_to_dense_slice_from_beginning(self):
        ''' Check that it works when the bounds start before/after the end'''
        k0 = [10, 11]
        k1 = [12, 13]
        k2 = [11, 12, 13, 14]

        v0 = [1, 2]
        v1 = [2, 3]
        v2 = [4, 5, 6, 7]

        keys = k0 + k1 + k2
        vals = v0 + v1 + v2

        bounds = [0, 2, 4, 7]

        arr = JaggedKeyValueArray(keys, vals, bounds)

        v, k = arr[1:].to_dense()

        expected_v = np.array([[0, 2, 3], [4, 5, 6]])
        expected_k = np.array([11, 12, 13])

        np.testing.assert_array_equal(expected_v, v)
        np.testing.assert_array_equal(expected_k, k)

    def test_to_dense_slice_to_end(self):
        ''' Check that it works when the bounds start before/after the end'''
        k0 = [10, 11, 12]
        k1 = [12, 13]
        k2 = [11, 12, 13, 14]

        v0 = [1, 2, 3]
        v1 = [2, 3]
        v2 = [4, 5, 6, 7]

        keys = k0 + k1 + k2
        vals = v0 + v1 + v2

        bounds = [0, 3, 5, 8]

        arr = JaggedKeyValueArray(keys, vals, bounds)

        v, k = arr[:2].to_dense()

        expected_v = np.array([[1, 2, 3, 0], [0, 0, 2, 3]])
        expected_k = np.array([10, 11, 12, 13])

        np.testing.assert_array_equal(expected_v, v)
        np.testing.assert_array_equal(expected_k, k)

    def test_get_utilized_keys(self):
        keys = [0, 1, 2, 3, 4, 5]
        vals = [10, 11, 12, 13, 15]

        bounds = [2, 4, 5]

        arr = JaggedKeyValueArray(keys, vals, bounds)

        result = arr.get_utilized_keys()
        expected = np.array([2, 3, 4])
        np.testing.assert_array_equal(expected, result)


class JaggedKeyValueArrayTests(unittest.TestCase):
    def setUp(self):
        self.k0 = [11, ]
        self.k1 = [12, 13]
        self.k2 = [11, 12, 13]

        self.v0 = [1, ]
        self.v1 = [2, 3]
        self.v2 = [4, 5, 6]

        self.keys = self.k0 + self.k1 + self.k2
        self.vals = self.v0 + self.v1 + self.v2

        self.bounds = [0, 1, 3, 6]

        self.arr = JaggedKeyValueArray(self.keys, self.vals, self.bounds)

    def test_to_fixed_depth(self):

        keys, values = self.arr.to_fixed_depth(1, reverse=True)

        expected_keys = np.array([[11], [13], [13]])
        expected_values = np.array([[1], [3], [6]])

        np.testing.assert_array_equal(expected_keys, keys)
        np.testing.assert_array_equal(expected_values, values)

    def test_to_fixed_depth2(self):

        keys, values = self.arr.to_fixed_depth(2, reverse=True)

        expected_keys = np.array([[11, NAN], [13, 12], [13, 12]])
        expected_values = np.array([[1, NAN], [3, 2], [6, 5]])

        np.testing.assert_array_equal(expected_keys, keys)
        np.testing.assert_array_equal(expected_values, values)

    def test_to_fixed_depth_reverse_False(self):

        keys, values = self.arr.to_fixed_depth(2, reverse=False)

        expected_keys = np.array([[11, NAN], [12, 13], [11, 12]])
        expected_values = np.array([[1, NAN], [2, 3], [4, 5]])

        np.testing.assert_array_equal(expected_keys, keys)
        np.testing.assert_array_equal(expected_values, values)

    def test_remove_values_smaller_than1(self):
        result = self.arr.remove_values_smaller_than(1.1)

        expected_keys = np.array([12, 13, 11, 12, 13])
        expected_values = np.array([2, 3, 4, 5, 6])

        expected_bounds = [0, 0, 2, 5]

        self.assertIsInstance(result, JaggedKeyValueArray)
        np.testing.assert_array_equal(expected_keys, result.keys)
        np.testing.assert_array_equal(expected_values, result.values)
        np.testing.assert_array_equal(expected_bounds, result.bounds)

    def test_remove_values_smaller_than2(self):
        result = self.arr.remove_values_smaller_than(3.1)

        expected_keys = np.array([11, 12, 13])
        expected_values = np.array([4, 5, 6])

        expected_bounds = [0, 0, 0, 3]

        self.assertIsInstance(result, JaggedKeyValueArray)
        np.testing.assert_array_equal(expected_keys, result.keys)
        np.testing.assert_array_equal(expected_values, result.values)
        np.testing.assert_array_equal(expected_bounds, result.bounds)

    def test_io(self):
        file = tempfile.TemporaryFile()

        self.arr.save(file)

        file.seek(0)
        arr2 = JaggedKeyValueArray.load(file)

        self.assertEqual(self.arr, arr2)
        file.close()

    def test_get_histogram(self):
        result = self.arr.get_histogram()
        expected = {
            11: 5,
            12: 7,
            13: 9
        }
        self.assertEqual(expected, result)

    def test___bool__(self):
        self.assertTrue(bool(self.arr))

    def test___bool__False(self):
        k = []
        v = []
        bounds = []
        arr = JaggedKeyValueArray(k, v, bounds)
        self.assertFalse(bool(arr))

    def test_to_dense_projection(self):
        projection = np.array([10, 11, 12, 13, 14])
        d = self.arr.to_dense_projection(projection)

        expected_d = np.array([[0, 1, 0, 0, 0],
                               [0, 0, 2, 3, 0],
                               [0, 4, 5, 6, 0]])

        print('d', d)
        np.testing.assert_array_equal(expected_d, d)

    def test_cumsum(self):
        dense, cols = self.arr.to_dense()
        cs = dense.cumsum(axis=0)
        e = JaggedKeyValueArray.from_dense(cs, cols)

        print(e)
        r = self.arr.cumsum()

        np.testing.assert_array_equal(e.bounds, r.bounds)
        np.testing.assert_array_equal(e.keys, r.keys)
        np.testing.assert_array_equal(e.values, r.values)

    def test_cumsum2(self):
        jkva = JaggedKeyValueArray(
            keys=[],
            values=[],
            bounds=[0, 0, 0, 0],
        )

        result = jkva.cumsum()
        print(result)

    def test_from_lists(self):
        key_list = [self.k0, self.k1, self.k2]
        val_list = [self.v0, self.v1, self.v2]

        result = JaggedKeyValueArray.from_lists(key_list, val_list)

        self.assertEqual(self.arr, result)

    def test_from_lists_empty(self):
        key_list = []
        val_list = []

        result = JaggedKeyValueArray.from_lists(key_list, val_list)

        self.assertEqual(0, len(result))

    def test_from_lists_float(self):
        key_list = [[1]]
        val_list = [[2]]

        result = JaggedKeyValueArray.from_lists(key_list, val_list)

        self.assertEqual(np.float32, result.keys.dtype)
        self.assertEqual(np.float32, result.values.dtype)

    def test___len__(self):
        self.assertEqual(3, len(self.arr))

    def test___getitem__row0(self):
        r = self.arr[0]

        e0 = np.array(self.k0)
        e1 = np.array(self.v0)

        np.testing.assert_array_equal(e0, r[0])
        np.testing.assert_array_equal(e1, r[1])

    def test___getitem__row1(self):
        r = self.arr[1]

        e0 = np.array(self.k1)
        e1 = np.array(self.v1)

        np.testing.assert_array_equal(e0, r[0])
        np.testing.assert_array_equal(e1, r[1])

    def test__getitem_rowneg1(self):
        r = self.arr[-1]

        e0 = np.array(self.k2)
        e1 = np.array(self.v2)

        np.testing.assert_array_equal(e0, r[0])
        np.testing.assert_array_equal(e1, r[1])

    def test__getitem_1dslice1(self):
        r = self.arr[:1]
        print('test__getitem_1dslice: %s' % r)
        self.assertIsInstance(r, JaggedKeyValueArray)
        self.assertEqual(1, len(r))
        self.assertEqual(0, r.bounds[0])
        self.assertEqual(1, r.bounds[1])

    def test__getitem_1dslice2(self):
        r = self.arr[1:3]
        print('test__getitem_1dslice: %s' % r)
        self.assertIsInstance(r, JaggedKeyValueArray)
        self.assertEqual(2, len(r))
        self.assertEqual(1, r.bounds[0])
        self.assertEqual(3, r.bounds[1])
        self.assertEqual(6, r.bounds[2])

    def test__getitem_1dslice_neg1(self):
        r = self.arr[-1:]
        print('test__getitem_1dslice_neg1: %s' % r)
        self.assertIsInstance(r, JaggedKeyValueArray)
        self.assertEqual(1, len(r))
        self.assertEqual(3, r.bounds[0])
        self.assertEqual(6, r.bounds[1])

    def test__getitem_1dslice_neg2(self):
        r = self.arr[-3:-1]
        print('test__getitem_1dslice_neg2: %s' % r)
        self.assertIsInstance(r, JaggedKeyValueArray)
        self.assertEqual(2, len(r))
        np.testing.assert_array_equal([0, 1, 3], r.bounds)

    def test___getitem__2dslice(self):
        k, v = self.arr[0, 0]
        self.assertEqual(11, k)
        self.assertEqual(1, v)

        k, v = self.arr[1, 0]
        self.assertEqual(12, k)
        self.assertEqual(2, v)

        k, v = self.arr[1, -1]
        self.assertEqual(13, k)
        self.assertEqual(3, v)

    def test___getitem__2dslice2(self):
        k, v = self.arr[0, :2]

        self.assertEqual(self.k0, k)
        self.assertEqual(self.v0, v)

        k, v = self.arr[1, :1]

        self.assertEqual(self.k1[:1], k)
        self.assertEqual(self.v1[:1], v)

        k, v = self.arr[2, 1:]

        np.testing.assert_array_equal(self.k2[1:], k)
        np.testing.assert_array_equal(self.v2[1:], v)

    def test___getitem__2dslice3(self):
        r = self.arr[:2, -2:]

        self.assertIsInstance(r, tuple)

        e0 = np.array([[0., 11.],
                       [12., 13.],
                       [12., 13.]])
        e1 = np.array([[0., 1.],
                       [2., 3.],
                       [5., 6.]])
        # print( r[1] )
        np.testing.assert_equal(e0, r[0])
        np.testing.assert_equal(e1, r[1])

    def test___getitem__2dslice4(self):
        k, v = self.arr[:, -1]

        expected_keys = np.array([11, 13, 13])
        expected_vals = np.array([1, 3, 6])

        np.testing.assert_equal(expected_keys, k)
        np.testing.assert_equal(expected_vals, v)

    def test_to_dense(self):
        data, cols = self.arr.to_dense()
        print(data)
        print(cols)

    def test_to_dense2(self):
        keys = [[0], [1, 2, 3], [2, 3]]
        values = [[10], [21, 22, 23], [32, 33]]

        arr = JaggedKeyValueArray.from_lists(keys, values)

        data, cols = arr.to_dense()
        print(data)
        print(cols)

    def test_from_dense(self):
        data = [[0, 1, 2],
                [3, 0, 4],
                [0, 5, 0]]
        cols = [10, 20, 30]

        data = np.array(data)
        cols = np.array(cols)
        r = JaggedKeyValueArray.from_dense(data, cols)
        print(r)

        e0 = np.array([1, 2, 3, 4, 5])
        e1 = np.array([20, 30, 10, 30, 20])
        e2 = np.array([0, 2, 4, 5])
        np.testing.assert_array_equal(e0, r.values)
        np.testing.assert_array_equal(e1, r.keys)
        np.testing.assert_array_equal(e2, r.bounds)

    def test_from_dense_nb(self):
        data = [[0, 1, 2],
                [3, 0, 4],
                [0, 5, 0]]
        cols = [10, 20, 30]

        data = np.array(data)
        cols = np.array(cols)
        r = JaggedKeyValueArray.from_dense_nb(data, cols)
        print(r)

        e0 = np.array([1, 2, 3, 4, 5])
        e1 = np.array([20, 30, 10, 30, 20])
        e2 = np.array([0, 2, 4, 5])
        np.testing.assert_array_equal(e0, r.values)
        np.testing.assert_array_equal(e1, r.keys)
        np.testing.assert_array_equal(e2, r.bounds)



if __name__ == '__main__':
    unittest.main()
