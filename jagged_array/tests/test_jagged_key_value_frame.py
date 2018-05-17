import numpy as np
import pandas as pd
import pandas.util.testing as tm
import tempfile
import unittest

from jagged_array.jagged_key_value_array import JaggedKeyValueArray
from jagged_array.jagged_key_value_series import JaggedKeyValueSeries
from jagged_array.jagged_key_value_frame import JaggedKeyValueFrame


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