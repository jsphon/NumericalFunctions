import numpy as np
import pandas as pd
import pandas.util.testing as tm
import tempfile
import unittest

from jagged_array.jagged_key_value_series import JaggedKeyValueSeries
from jagged_array.jagged_key_value_array import JaggedKeyValueArray
import jagged_array.jagged_key_value_series as mod


class ModuleTests(unittest.TestCase):
    def test_floor_to_nearest(self):
        x = np.array([-10, -6, -5, -3, 0, 1, 5, 9, 10, 11])
        expected = np.array([-10, -10, -5, -5, 0, 0, 5, 5, 10, 10])

        result = mod.floor_to_nearest_int(x, 5)
        np.testing.assert_array_equal(expected, result)

    def test_get_resampled_index(self):
        x = np.array([-10, -6, -5, -3, 0, 1, 5, 9, 10, 11])

        result = mod.get_resampled_index(x, 5)

        expected = np.array([-10, -5, 0, 5, 10])
        np.testing.assert_array_equal(expected, result)

    def test_get_resample_indices(self):
        x = np.array([-10, -6, -5, -3, 0, 1, 5, 9, 10, 11])
        expected = np.array([0, 2, 4, 6, 8])
        result = mod.get_resample_indices(x, 5)

        np.testing.assert_array_equal(expected, result)


def get_empty_jagged_key_value_series():
    arr = JaggedKeyValueArray([], [], [0])
    index = []
    return JaggedKeyValueSeries(arr, index)


class LongerJaggedKeyValueSeriesTests(unittest.TestCase):
    def setUp(self):
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
        index = [-5, -3, 0, 3, 5, 7, 9]
        self.s = JaggedKeyValueSeries(arr, index)

    def test___repr__(self):
        result = str(self.s)
        print(result)


class JaggedKeyValueSeriesTests(unittest.TestCase):
    def setUp(self):

        k0 = [10, 11]
        k1 = [11, 12, 13]
        k2 = [12, 13]
        k3 = []
        k4 = [14]

        v0 = [1, 2]
        v1 = [3, 4, 5]
        v2 = [6, 7]
        v3 = []
        v4 = [8]

        keys = k0 + k1 + k2 + k3 + k4
        vals = v0 + v1 + v2 + v3 + v4
        bounds = [0, 2, 5, 7, 7, 8]
        arr = JaggedKeyValueArray(keys, vals, bounds)
        index = [-5, -3, 0, 3, 5]
        self.s = JaggedKeyValueSeries(arr, index)

    def test_to_fixed_depth(self):
        """
        For presenting data as a market depth dataframe
        :return:
        """

        # TODO: Finish this later

        columns = pd.MultiIndex.from_tuples(
            [
                ('key', 0),
                ('key', 1),
                ('key', 2),
                ('value', 0),
                ('value', 1),
                ('value', 2),
                    ],
            names=['type', 'level']
        )
        nan = np.nan
        data = [
            [11, 10, nan, 2, 1, nan],
            [13, 12, 11,  5, 4, 3],
            [13, 12, nan, 7, 6, nan],
            [nan, nan, nan, nan, nan, nan],
            [14, nan, nan, 8, nan, nan]
        ]

        expected = pd.DataFrame(
            columns=columns,
            data = data,
            index = self.s.index
        )

        result = self.s.to_fixed_depth(3, reverse=True)

        pd.testing.assert_frame_equal(expected, result, check_dtype=False)

    def test_get_ohlvc_frame(self):

        df = self.s.get_ohlcv_frame(1)

        expected = pd.DataFrame([
            [10, 11, 10, 10, 3],
            [12, 13, 11, 12, 12],
            [12, 13, 12, 12, 13],
            [np.nan, np.nan, np.nan, np.nan, 0],
            [14, 14, 14, 14, 8]
        ],
            columns=['o', 'h', 'l', 'c', 'v'],
            index=[-5, -3, 0, 3, 5])

        tm.assert_frame_equal(expected, df, check_dtype=False)

    def test_load_save(self):

        with tempfile.TemporaryFile() as f:
            self.s.save(f)

            f.seek(0)

            s2 = JaggedKeyValueSeries.load(f)

        self.assertEqual(self.s, s2)

    def test_resample(self):

        result = self.s.resample(5)

        expected = JaggedKeyValueSeries(
            keys=[10, 11, 12, 13, 12, 13, 14],
            values=[1, 5, 4, 5, 6, 7, 8],
            bounds=[0, 4, 6, 7],
            index=[-5, 0, 5],
        )
        self.assertEqual(expected, result)

    def test_ravel(self):

        index, keys, values = self.s.ravel()

        np.testing.assert_array_equal([-5, -5, -3, -3, -3, 0, 0, 5], index)
        np.testing.assert_array_equal([10, 11, 11, 12, 13, 12, 13, 14], keys)
        np.testing.assert_array_equal([1, 2, 3, 4, 5, 6, 7, 8], values)

    def test_get_v(self):

        result = self.s.get_v(5)

        expected = np.array([15, 13, 8])

        np.testing.assert_array_equal(expected, result)

    def test_get_ohlc(self):

        result = self.s.get_ohlc(5)

        expected = np.array([
            [10, 13, 10, 12],
            [12, 13, 12, np.nan],
            [14, 14, 14, 14],
        ])

        np.testing.assert_array_equal(expected, result)

    def test_get_resample_index_bounds(self):
        result = self.s.get_resample_index_bounds(5)
        expected = np.array([
            [0, 2, 2, 5],
            [5, 7, 7, 7],
            [7, 8, 7, 8],
        ])

        np.testing.assert_array_equal(expected, result)

    def test_bool_true(self):
        self.assertTrue(self.s)

    def test_bool_false(self):
        self.assertFalse(get_empty_jagged_key_value_series())

    def test___getitem__int(self):

        for i, k, v in (
                (-5, [10, 11], [1, 2]),
                (-3, [11, 12, 13], [3, 4, 5]),
                (0, [12, 13], [6, 7]),
                (3, [], []),
                (5, [14], [8]),
        ):
            result = self.s[i]
            np.testing.assert_array_equal(k, result[0])
            np.testing.assert_array_equal(v, result[1])

    def test__getitem__2ints(self):
        for cfg in (
                {
                    'i0': -5,
                    'i1': 0,
                    'expected': JaggedKeyValueSeries(
                        keys=[10, 11, 11, 12, 13],
                        values=[1, 2, 3, 4, 5],
                        bounds=[0, 2, 5],
                        index=[-5, -3],
                    )
                },
                {
                    'i0': 0,
                    'i1': None,
                    'expected': JaggedKeyValueSeries(
                        keys=[12, 13, 14],
                        values=[6, 7, 8],
                        bounds=[0, 2, 2, 3],
                        index=[0, 3, 5],
                    )
                },
                {
                    'i0': None,
                    'i1': 0,
                    'expected': JaggedKeyValueSeries(
                        keys=[10, 11, 11, 12, 13],
                        values=[1, 2, 3, 4, 5],
                        bounds=[0, 2, 5],
                        index=[-5, -3],
                    )
                },
        ):
            expected = cfg['expected']
            result = self.s[cfg['i0']:cfg['i1']]

            self.assertEqual(expected, result)

    def test_cumsum(self):
        """
        Assert that cumsum works, though we only slice on the
        first 2 levels in order to keep it short.

        Remember that s[:0] will include indices -5 and -3, but not 0
        :return:
        """
        result = self.s[:0].cumsum()

        expected = JaggedKeyValueSeries(
            keys=[10, 11, 10, 11, 12, 13, ],
            values=[1, 2, 1, 5, 4, 5],
            bounds=[0, 2, 6],
            index=[-5, -3],
        )

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
