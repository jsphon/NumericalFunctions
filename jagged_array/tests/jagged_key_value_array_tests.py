import numpy as np
import pandas as pd
import tempfile
import unittest

from jagged_array.jagged_key_value_array import JaggedKeyValueArray
import jagged_array.jagged_key_value_array as mod


class ModuleTests(unittest.TestCase):

    def test_floor_to_nearest(self):

        x = np.array([-10, -6, -5, -3, 0, 1, 5, 9, 10, 11])
        expected = np.array([-10, -10, -5, -5, 0, 0, 5, 5, 10, 10])

        result = mod.floor_to_nearest_int(x, 5)
        np.testing.assert_array_equal(expected, result)

    def test_get_resample_indices(self):
        x = np.array([-10, -6, -5, -3, 0, 1, 5, 9, 10, 11])
        expected = np.array([0, 2, 4, 6, 8])
        result = mod.get_resample_indices(x, multiplier=5)

        np.testing.assert_array_equal(expected, result)


class MoreJaggedKeyValueArrayTests(unittest.TestCase):

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

    def test_ravel(self):
        expected_index = np.array([0, 1, 1, 2, 2, 2])

        expected_keys = self.keys
        expected_vals = self.vals

        i, k, v = self.arr.ravel()

        np.testing.assert_array_equal(expected_index, i)
        np.testing.assert_array_equal(expected_keys, k)
        np.testing.assert_array_equal(expected_vals, v)

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


class OHLCTests2WithDateIndex(unittest.TestCase):
    def setUp(self):
        self.k0 = [11, ]  # 4
        self.k1 = [12, 13]  # 6
        self.k2 = [13, 14, 15]  # 8

        self.v0 = [1, ]
        self.v1 = [2, 3]
        self.v2 = [4, 5, 6]

        self.keys = self.k0 + self.k1 + self.k2
        self.vals = self.v0 + self.v1 + self.v2

        self.bounds = [0, 1, 3, 6]

        #self.index = pd.date_range('2018-01-01 00:00:00', freq='5s', periods=3)
        self.index = [0, 5, 10]#pd.date_range('2018-01-01 00:00:00', freq='5s', periods=3)
        self.arr = JaggedKeyValueArray(
            self.keys,
            self.vals,
            self.bounds,
            index=self.index
        )

    def test_get_ohlcv_frame_by_interval(self):
        result = self.arr.get_ohlcv_frame_by_interval(5)
        print('test_get_ohlc_by_interval: %s' % str(result))
        #expected_index = pd.date_range('2018-01-01 00:00:00', freq='5s', periods=3)
        expected_index = [0, 1, 2]
        expected_values = [[11.0, 11.0, 11.0, 11.0, 1.0],
                           [12.0, 13.0, 12.0, 12.0, 5.0],
                           [14.0, 15.0, 13.0, 14.0, 15.0]]

        np.testing.assert_array_equal(expected_values, result.values)
        np.testing.assert_array_equal(['o', 'h', 'l', 'c', 'v'], result.columns)
        np.testing.assert_array_equal(expected_index, result.index)

    def xtest_get_ohlcv_frame_by_date_index(self):
        result = self.arr.get_ohlcv_frame_by_date_index('5s')
        print(result)

        expected_index = pd.date_range('2018-01-01 00:00:00', freq='5s', periods=3)
        expected_values = [[11.0, 11.0, 11.0, 11.0, 1.0],
                           [12.0, 13.0, 12.0, 12.0, 5.0],
                           [14.0, 15.0, 13.0, 14.0, 15.0]]

        np.testing.assert_array_equal(expected_values, result.values)
        np.testing.assert_array_equal(['o', 'h', 'l', 'c', 'v'], result.columns)
        np.testing.assert_array_equal(expected_index, result.index)

    def xtest_get_resample_index_bounds(self):
        result = self.arr.get_resample_index_bounds('5s')

        np.testing.assert_array_equal(
            np.array([0, 1, 3]),
            result[:, 0]
        )

        np.testing.assert_array_equal(
            np.array([1, 3, 6]),
            result[:, 1]
        )

        np.testing.assert_array_equal(
            np.array([0, 1, 3]),
            result[:, 2]
        )

        np.testing.assert_array_equal(
            np.array([1, 3, 6]),
            result[:, 3]
        )

class OHLCTests3WithNumericalIndex(unittest.TestCase):
    def setUp(self):
        self.k0 = [11]
        self.k1 = []
        self.k2 = [13, 14, 15]

        self.v0 = [1, ]
        self.v1 = []
        self.v2 = [4, 5, 6]

        self.keys = self.k0 + self.k1 + self.k2
        self.vals = self.v0 + self.v1 + self.v2

        self.bounds = [0, 1, 1, 4]

        self.index = pd.date_range('2018-01-01 00:00:00', freq='5s', periods=3)
        self.index = [0, 5, 10]
        self.arr = JaggedKeyValueArray(
            self.keys,
            self.vals,
            self.bounds,
            index=self.index
        )

    def test_get_resample_index_bounds(self):
        result = self.arr.get_resample_index_bounds(5)
        expected = np.array([
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 4, 1, 4],
        ])

        np.testing.assert_array_equal(expected, result)

    def test_get_ohlcv_frame(self):
        result = self.arr.get_ohlcv_frame_by_interval(5)
        print(result)

        expected_index = [0, 1, 2]#pd.date_range('2018-01-01 00:00:00', freq='5s', periods=3)
        expected_values = [[11.0, 11.0, 11.0, 11.0, 1.0],
                           [np.nan, np.nan, np.nan, np.nan, 0.0],
                           [14.0, 15.0, 13.0, 14.0, 15.0]]

        np.testing.assert_array_equal(expected_values, result.values)
        np.testing.assert_array_equal(['o', 'h', 'l', 'c', 'v'], result.columns)
        np.testing.assert_array_equal(expected_index, result.index)


# class OHLCTests3(unittest.TestCase):
#     def setUp(self):
#         self.k0 = [11]
#         self.k1 = []
#         self.k2 = [13, 14, 15]
#
#         self.v0 = [1, ]
#         self.v1 = []
#         self.v2 = [4, 5, 6]
#
#         self.keys = self.k0 + self.k1 + self.k2
#         self.vals = self.v0 + self.v1 + self.v2
#
#         self.bounds = [0, 1, 1, 4]
#
#         self.index = pd.date_range('2018-01-01 00:00:00', freq='5s', periods=3)
#         self.arr = JaggedKeyValueArray(
#             self.keys,
#             self.vals,
#             self.bounds,
#             index=self.index
#         )
#
#     def test_get_resample_index_bounds(self):
#         result = self.arr.get_resample_index_bounds('5s')
#         expected = np.array([
#             [0, 1, 0, 1],
#             [1, 1, 1, 1],
#             [1, 5, 1, 5],
#         ])
#
#         np.testing.assert_array_equal(expected, result)
#
#     def test_get_ohlcv_frame(self):
#         result = self.arr.get_ohlcv_frame('5s')
#         print(result)
#
#         expected_index = pd.date_range('2018-01-01 00:00:00', freq='5s', periods=3)
#         expected_values = [[11.0, 11.0, 11.0, 11.0, 1.0],
#                            [np.nan, np.nan, np.nan, np.nan, 0.0],
#                            [14.0, 15.0, 13.0, 14.0, 15.0]]
#
#         np.testing.assert_array_equal(expected_values, result.values)
#         np.testing.assert_array_equal(['o', 'h', 'l', 'c', 'v'], result.columns)
#         np.testing.assert_array_equal(expected_index, result.index)


# class OHLCTests3(unittest.TestCase):
#     def setUp(self):
#         self.keys = [580, 590]  # , 600]
#         self.vals = [1, 2]  # , 2]
#
#         self.bounds = [0, 2, 2]
#
#         self.index = pd.date_range('2018-01-01 00:00:00', freq='5s', periods=2)
#         self.arr = JaggedKeyValueArray(
#             self.keys,
#             self.vals,
#             self.bounds,
#             index=self.index
#         )
#
#     def test_get_resample_index_bounds(self):
#         result = self.arr.get_resample_index_bounds('5s')
#
#         expected = np.array([
#             [0, 2, 0, 2],
#             [2, 2, 2, 2],  # empty
#         ])
#         np.testing.assert_array_equal(expected, result)
#
#     def test_get_ohlcv_frame(self):
#         result = self.arr.get_ohlcv_frame('5s')
#         print(result)
#
#         expected_index = pd.date_range('2018-01-01 00:00:00', freq='5s', periods=2)
#         expected_values = [[580.0, 590.0, 580.0, 580.0, 3.0],
#                            [np.nan, np.nan, np.nan, np.nan, 0.0]]
#
#         np.testing.assert_array_equal(expected_values, result.values)
#         np.testing.assert_array_equal(['o', 'h', 'l', 'c', 'v'], result.columns)
#         np.testing.assert_array_equal(expected_index, result.index)
#
#     def test_get_v(self):
#         v = self.arr.get_v('5s')
#
#         expected = np.array([3, 0])
#         np.testing.assert_array_equal(expected, v)


class OHLCTests(unittest.TestCase):
    def setUp(self):
        self.k0 = [11, ]  # 4
        self.k1 = [12, 13]  # 6
        self.k2 = [14, 15, 16]  # 8
        self.k3 = [17, 18, 19]  # 10
        self.k4 = [20, 21, 22]  # 12
        self.k5 = [23, 24, 25]  # 14

        self.v0 = [1, ]
        self.v1 = [2, 3]
        self.v2 = [4, 5, 6]
        self.v3 = [7, 8, 9]
        self.v4 = [10, 11, 13]
        self.v5 = [14, 15, 16]

        self.keys = self.k0 + self.k1 + self.k2 + self.k3 + self.k4 + self.k5  # + self.k6
        self.vals = self.v0 + self.v1 + self.v2 + self.v3 + self.v4 + self.v5  # + self.v6

        self.bounds = [0, 1, 3, 6, 9, 12, 15]

        self.date_index = pd.date_range('2018-01-01 00:00:04', freq='2s', periods=6)
        self.index = [4, 6, 8, 10, 12, 14]
        self.arr = JaggedKeyValueArray(
            self.keys,
            self.vals,
            self.bounds,
            index=self.index,
            date_index = self.date_index
        )

    def test_get_ohlc_by_interval(self):
        result = self.arr.get_ohlc_by_interval(5)
        expected = np.array([
            [11, 11, 11, 11],
            [12, 16, 12, 15],
            [18, 25, 17, 24],
        ])

        print(result)
        np.testing.assert_array_equal(expected, result)

    def test_get_ohlc_by_date_index(self):
        result = self.arr.get_ohlc_by_date_index('5s')
        expected = np.array([
            [11, 11, 11, 11],
            [12, 16, 12, 15],
            [18, 25, 17, 24],
        ])

        print(result)
        np.testing.assert_array_equal(expected, result)

    def test_get_resample_index_bounds(self):
        result = self.arr.get_resample_index_bounds(5)

        expected = np.array([
            [0, 1, 0, 1],
            [1, 3, 3, 6],
            [6, 9, 12, 15],
        ])
        np.testing.assert_array_equal(expected, result)

    def test_get_resample_date_index_bounds(self):
        result = self.arr.get_resample_date_index_bounds('5s')

        expected = np.array([
            [0, 1, 0, 1],
            [1, 3, 3, 6],
            [6, 9, 12, 15],
        ])
        np.testing.assert_array_equal(expected, result)

    def xtest_get_ohlcv_frame_by_date_index(self):
        result = self.arr.get_ohlcv_frame_by_date_index('5s')
        print(result)

        expected_index = pd.date_range('2018-01-01 00:00:00', freq='5s', periods=3)
        expected_values = [[11.0, 11.0, 11.0, 11.0, 1.0],
                           [12.0, 16.0, 12.0, 15.0, 20.0],
                           [18.0, 25.0, 17.0, 24.0, 103.0]]

        np.testing.assert_array_equal(expected_values, result.values)
        np.testing.assert_array_equal(['o', 'h', 'l', 'c', 'v'], result.columns)
        np.testing.assert_array_equal(expected_index, result.index)

    def test_get_v_by_index(self):
        result = self.arr.get_v_by_index(5)
        expected = np.array([1, 20, 103])

        np.testing.assert_array_equal(expected, result)

    #TODO: Fix the date_index stuff later
    def xtest_get_v_by_date_index(self):
        result = self.arr.get_v_by_date_index('5s')
        expected = np.array([1, 20, 103])

        np.testing.assert_array_equal(expected, result)


class JaggedKeyValueArrayWithDateTimeIndexTests(unittest.TestCase):

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

        self.index = np.array([3, 4, 5])

        self.arr = JaggedKeyValueArray(
            self.keys,
            self.vals,
            self.bounds,
            index=self.index
        )

    def test_ravel(self):

        expected_index = np.array([
            self.arr.index[0],
            self.arr.index[1],
            self.arr.index[1],
            self.arr.index[2],
            self.arr.index[2],
            self.arr.index[2],
        ], dtype=self.arr.index.dtype)

        expected_keys = self.keys
        expected_vals = self.vals

        i, k, v = self.arr.ravel()

        np.testing.assert_array_equal(expected_index, i)
        np.testing.assert_array_equal(expected_keys, k)
        np.testing.assert_array_equal(expected_vals, v)

    # TODO: We will need slicing on both the index, and the
    # actual row location. i.e. If we have floating point index
    # values, but want to get it between integer points, will out
    # slicing handle that?
    # def test_get_between(self):
    #     d0 = self.index[0]
    #     d1 = self.index[1]
    #     d2 = self.index[2]
    #
    #     result = self.arr.get_between(d0, d2)
    #
    #     expected_index = pd.DatetimeIndex([d1])
    #     self.assertEqual(expected_index, result.index)
    #     np.testing.assert_array_equal(self.k1, result[0][0])
    #     np.testing.assert_array_equal(self.v1, result[0][1])

    #def test___getitem__row0(self):
    def test_loc(self):

        for index, expected in (
            [3, ([11], [1])],
            [4, ([12, 13], [2, 3])],
            [5, ([11, 12, 13], [4, 5, 6])],
        ):

            r = self.arr.loc(index)

            expected_keys = np.array(expected[0])
            expected_values = np.array(expected[1])
            np.testing.assert_array_equal(expected_keys, r[0])
            np.testing.assert_array_equal(expected_values, r[1])

    def test_loc_slice(self):

        for iStart, iEnd, b, k, v, ind in (
            [3, 4, [0, 1], [11], [1], [3]],
            [4, 5, [0, 2], [12, 13], [2, 3], [4]],
            [3, 5, [0, 1, 3], [11, 12, 13], [1, 2, 3], [3, 4]],
        ):
            r = self.arr.loc_slice(iStart, iEnd)
            self.assertIsInstance(r, JaggedKeyValueArray)

            expected_bounds = np.array(b)
            np.testing.assert_array_equal(expected_bounds, r.bounds)

            expected_keys = np.array(k)
            np.testing.assert_array_equal(expected_keys, r.keys)

            expected_values = np.array(v)
            np.testing.assert_array_equal(expected_values, r.values)

            expected_index = np.array(ind)
            np.testing.assert_array_equal(expected_index, r.index)

    def test_loc_slice_no_first(self):

        for last, b, k, v, ind in (
                [4, [0, 1], [11], [1], [3]],
                [5, [0, 1, 3], [11, 12, 13], [1, 2, 3], [3, 4]],
                [6, [0, 1, 3, 6], [11, 12, 13, 11, 12, 13], [1, 2, 3, 4, 5, 6], [3, 4, 5]],
        ):
            r = self.arr.loc_slice(last=last)
            self.assertIsInstance(r, JaggedKeyValueArray)

            expected_bounds = np.array(b)
            np.testing.assert_array_equal(expected_bounds, r.bounds)

            expected_keys = np.array(k)
            np.testing.assert_array_equal(expected_keys, r.keys)

            expected_values = np.array(v)
            np.testing.assert_array_equal(expected_values, r.values)

            expected_index = np.array(ind)
            np.testing.assert_array_equal(expected_index, r.index)

    def test_loc_slice_no_last(self):

        for first, b, k, v, ind in (
                [3, [0, 1, 3, 6], [11, 12, 13, 11, 12, 13], [1, 2, 3, 4, 5, 6], [3, 4, 5]],
                [4, [0, 2, 5], [12, 13, 11, 12, 13], [2, 3, 4, 5, 6], [4, 5]],
                [5, [0, 3], [11, 12, 13], [4, 5, 6], [5]],
        ):
            r = self.arr.loc_slice(first=first)
            self.assertIsInstance(r, JaggedKeyValueArray)

            expected_bounds = np.array(b)
            np.testing.assert_array_equal(expected_bounds, r.bounds)

            expected_keys = np.array(k)
            np.testing.assert_array_equal(expected_keys, r.keys)

            expected_values = np.array(v)
            np.testing.assert_array_equal(expected_values, r.values)

            expected_index = np.array(ind)
            np.testing.assert_array_equal(expected_index, r.index)

    def test_get_resampled_index(self):
        date_range = pd.date_range('20180101', freq='1s', periods=11)
        print('date_range is %s' % date_range)

        result = mod.get_resampled_datetime_index(date_range, freq='5s')

        expected = pd.date_range('20180101', freq='5s', periods=3)

        np.testing.assert_array_equal(expected, result)

    def test_get_resampled_index2(self):
        """
        If resampled to the same frequency, assert that it remains unchanged
        """

        date_range = pd.date_range('20180101', freq='5s', periods=11)
        result = mod.get_resampled_datetime_index(date_range, freq='5s')
        np.testing.assert_array_equal(date_range, result)

    def test_resample(self):
        k0 = [11, 12, 13]
        k2 = [11, 12, 13]

        v0 = [1, 2, 3]
        v2 = [4, 5, 6]

        expected = JaggedKeyValueArray.from_lists([k0, k2], [v0, v2])

        result = self.arr.resample(5)

        np.testing.assert_array_equal(expected.keys, result.keys)
        np.testing.assert_array_equal(expected.values, result.values)
        np.testing.assert_array_equal(expected.bounds, result.bounds)
        np.testing.assert_array_equal(expected.index, result.index)

#
# class JaggedKeyValueArrayWithDateTimeIndexTests(unittest.TestCase):
#
#     def setUp(self):
#         self.k0 = [11, ]
#         self.k1 = []
#         self.k2 = [11, 12, 13]
#
#         self.v0 = [1, ]
#         self.v1 = []
#         self.v2 = [4, 5, 6]
#
#         self.keys = self.k0 + self.k1 + self.k2
#         self.vals = self.v0 + self.v1 + self.v2
#
#         self.bounds = [0, 1, 1, 4]
#
#         self.index = np.array([3, 4, 5])
#
#         self.arr = JaggedKeyValueArray(
#             self.keys,
#             self.vals,
#             self.bounds,
#             index=self.index
#         )
#
#       Might be unneeded. We'll just get zeros
#     def test_resample_with_gap(self):
#         """
#         Write a test to show that resample works
#         if there is a gap in the resampled index
#         :return:
#         """
#
#         k0 = [11]
#         k1 = []
#         k2 = [11, 12, 13]
#
#         v0 = [1]
#         v1 = []
#         v2 = [4, 5, 6]
#
#         expected = JaggedKeyValueArray.from_lists([k0, k1, k2], [v0, v1, v2])
#
#         result = self.arr.resample(5)
#
#         np.testing.assert_array_equal(expected.keys, result.keys)
#         np.testing.assert_array_equal(expected.values, result.values)
#         np.testing.assert_array_equal(expected.bounds, result.bounds)
#         np.testing.assert_array_equal(expected.index, result.index)
