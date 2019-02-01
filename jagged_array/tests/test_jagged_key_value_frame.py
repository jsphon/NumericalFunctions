import numpy as np
import pandas as pd
import pandas.util.testing as tm
import unittest

from jagged_array.jagged_key_value_array import JaggedKeyValueArray
from jagged_array.jagged_key_value_series import JaggedKeyValueSeries
from jagged_array.jagged_key_value_frame import JaggedKeyValueFrame


NAN = np.nan


class JaggedKeyValueFrameTests(unittest.TestCase):

    def setUp(self):
        data = {
            123: JaggedKeyValueArray(
                keys=[2, 3, 4],
                values=[1, 2, 3],
                bounds=[0, 1, 2, 3],
            )
            ,
            567: JaggedKeyValueArray(
                keys=[4, 5, 6],
                values=[3, 2, 1],
                bounds=[0, 1, 2, 3],
            ),
            789: JaggedKeyValueArray(
                keys=[],
                values=[],
                bounds=[0, 0, 0, 0],
            ),
        }

        index = [10, 11, 12]

        self.jf = JaggedKeyValueFrame(data, index)

    def test_remove_values_smaller_than(self):

        expected_data = {
            123: JaggedKeyValueArray(
                keys=[3, 4],
                values=[2, 3],
                bounds=[0, 0, 1, 2],
            )
            ,
            567: JaggedKeyValueArray(
                keys=[4, 5],
                values=[3, 2],
                bounds=[0, 1, 2, 2],
            ),
            789: JaggedKeyValueArray(
                keys=[],
                values=[],
                bounds=[0, 0, 0, 0],
            ),
        }

        expected = JaggedKeyValueFrame(expected_data, [10, 11, 12])

        result = self.jf.remove_values_smaller_than(1.1)
        #print(expected[123])
        print(type(result[123]))
        print(result[123].arr.keys)
        print(result[123].arr.values)
        print(result[123].arr.bounds)
        self.assertEqual(expected, result)

    def test_cumsum(self):

        expected_data = {
            123: JaggedKeyValueArray(
                keys=[2, 2, 3, 2, 3, 4],
                values=[1, 1, 2, 1, 2, 3],
                bounds=[0, 1, 3, 6],
            )
            ,
            567: JaggedKeyValueArray(
                keys=[4, 4, 5, 4, 5, 6],
                values=[3, 3, 2,3, 2, 1],
                bounds=[0, 1, 3, 6],
            ),
            789: JaggedKeyValueArray(
                keys=[],
                values=[],
                bounds=[0, 0, 0, 0],
            ),
        }

        expected = JaggedKeyValueFrame(expected_data, index=[10, 11, 12])

        result = self.jf.cumsum()

        self.assertEqual(expected, result)

    def test_get_fixed_depth_frame(self):

        result = self.jf.get_fixed_depth_frame(depth=2, reverse=False)

        data = [
            [2, NAN, 1, NAN, 4, NAN, 3, NAN, NAN, NAN, NAN, NAN],
            [3, NAN, 2, NAN, 5, NAN, 2, NAN, NAN, NAN, NAN, NAN],
            [4, NAN, 3, NAN, 6, NAN, 1, NAN, NAN, NAN, NAN, NAN],

        ]

        columns = pd.MultiIndex.from_tuples([
            (123, 'key', 0),
            (123, 'key', 1),
            (123, 'value', 0),
            (123, 'value', 1),
            (567, 'key', 0),
            (567, 'key', 1),
            (567, 'value', 0),
            (567, 'value', 1),
            (789, 'key', 0),
            (789, 'key', 1),
            (789, 'value', 0),
            (789, 'value', 1),
        ], names=[None, 'type', 'level'])

        expected = pd.DataFrame(data, columns=columns, index=self.jf.index)

        pd.testing.assert_frame_equal(expected, result, check_dtype=False)

    def test_row_slice(self):
        result = self.jf.row_slice(11, 12)

        expected_data = {
            123: JaggedKeyValueArray(
                keys=[3, 4],
                values=[2, 3],
                bounds=[0, 1, 2],
            )
            ,
            567: JaggedKeyValueArray(
                keys=[5, 6],
                values=[2, 1],
                bounds=[0, 1, 2],
            )
            ,
            789: JaggedKeyValueArray(
                keys=[],
                values=[],
                bounds=[0, 0, 0],
            )
        }
        expected_index = [10, 11]
        expected = JaggedKeyValueFrame(expected_data, expected_index)

        self.assertTrue(expected==result)

    def test___getitem__single(self):

        result = self.jf[123]
        self.assertIsInstance(result, JaggedKeyValueSeries)

    def test_get_item__list(self):

        result = self.jf[[123, 567]]
        self.assertIsInstance(result, JaggedKeyValueFrame)

    def test_get_ohlcv_frame(self):

        expected = pd.DataFrame(
            [
            [2, 2, 2, 2, 1, 4, 4, 4, 4, 3, np.nan, np.nan, np.nan, np.nan, 0],
            [3, 3, 3, 3, 2, 5, 5, 5, 5, 2, np.nan, np.nan, np.nan, np.nan, 0],
            [4, 4, 4, 4, 3, 6, 6, 6, 6, 1, np.nan, np.nan, np.nan, np.nan, 0],
            ],
            index = [10, 11, 12],
        )
        columns = pd.MultiIndex.from_tuples(
            (
            (123, 'o'),
            (123, 'h'),
            (123, 'l'),
            (123, 'c'),
            (123, 'v'),
            (567, 'o'),
            (567, 'h'),
            (567, 'l'),
            (567, 'c'),
            (567, 'v'),
            (789, 'o'),
            (789, 'h'),
            (789, 'l'),
            (789, 'c'),
            (789, 'v'),
            )
        )
        expected.columns = columns

        df = self.jf.get_ohlcv_frame(1)

        tm.assert_frame_equal(expected, df, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
