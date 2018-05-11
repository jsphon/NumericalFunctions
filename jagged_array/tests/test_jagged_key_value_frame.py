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
            )
        }

        index = [10, 11, 12]

        self.jf = JaggedKeyValueFrame(data, index)

    def test___getitem__single(self):

        result = self.jf[123]
        self.assertIsInstance(result, JaggedKeyValueSeries)

    def test_get_item__list(self):

        result = self.jf[[123, 567]]
        self.assertIsInstance(result, JaggedKeyValueFrame)

    def test_get_ohlcv_frame(self):

        expected = pd.DataFrame(
            [
            [2, 2, 2, 2, 1, 4, 4, 4, 4, 3],
            [3, 3, 3, 3, 2, 5, 5, 5, 5, 2],
            [4, 4, 4, 4, 3, 6, 6, 6, 6, 1],
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
            )
        )
        expected.columns = columns

        print(expected)
        df = self.jf.get_ohlcv_frame(1)
        print(df)
        tm.assert_frame_equal(expected, df, check_dtype=False)