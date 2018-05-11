import numpy as np
import pandas as pd
import tempfile
import unittest

from jagged_array.jagged_key_value_series import JaggedKeyValueSeries
from jagged_array.jagged_key_value_array import JaggedKeyValueArray
from jagged_array.jagged_key_value_array import JaggedKeyValueArray
from jagged_array.jagged_key_value_frame import JaggedKeyValueFrame


class JaggedKeyValueFrameTests(unittest.TestCase):
    def test_xxx(self):
        data = {
            '123': JaggedKeyValueArray(
                keys=[200.0, 300.0, 400.0],
                values=[1.0, 1.0, 1.0],
                bounds=[0, 1, 2, 3],
            )
            ,
            '567': JaggedKeyValueArray(
                keys=[400.0, 500.0, 600.0],
                values=[1.0, 1.0, 1.0],
                bounds=[0, 1, 2, 3],
            )
        }

        index = [10, 11, 12]

        jf = JaggedKeyValueFrame(data, index)
