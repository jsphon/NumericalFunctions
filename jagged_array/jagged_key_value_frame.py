from collections import OrderedDict

import pandas as pd
import numpy as np

from jagged_array.lib import is_sorted
from jagged_array.jagged_key_value_array import JaggedKeyValueArray
from jagged_array.jagged_key_value_series import JaggedKeyValueSeries

COLUMN_TYPES = (int, np.int, np.int64, str)


class JaggedKeyValueFrame(object):
    def __init__(self, arrs=None, index=None):
        """

        :param arrs dict of JaggedKeyValueArrays:
        :param index shared by all arrays
        :param columns: labels for each array
        """

        self.arrs = arrs
        self.index = index

        if index is None:
            first_arr_name = sorted(arrs.keys())[0]
            first_arr = arrs[first_arr_name]
            self.index = np.arange(len(first_arr))
        elif isinstance(index, (list, tuple)):
            self.index = np.array(index)

        self._verify()

    def __eq__(self, other):
        if self.arrs.keys() != other.arrs.keys():
            return False
        for k, arr in self.arrs.items():
            assert arr == other.arrs[k], 'series %s are different' % k

        return True
        # for arr in self.arrs

    def _verify(self):
        for k, arr in self.arrs.items():
            assert isinstance(arr, JaggedKeyValueArray), 'arr is a %s' % type(arr)
            assert len(arr) == len(self.index), '%s!=%s' % (len(arr), len(self.index))
        assert is_sorted(self.index), 'index should be sorted'

    def __getitem__(self, i):

        if isinstance(i, COLUMN_TYPES):
            arr = self.arrs[i]
            return JaggedKeyValueSeries(arr=arr, index=self.index)
        elif isinstance(i, list):
            arrs = dict((k, self.arrs[k]) for k in i)
            return JaggedKeyValueFrame(arrs, index=self.index)

    def row_slice(self, istart, iend):
        i0 = np.searchsorted(self.index, istart)
        i1 = np.searchsorted(self.index, iend, side='right')
        arrs = OrderedDict()
        for k, arr in self.arrs.items():
            arrs[k] = arr[i0:i1]
        new_index = self.index[i0:i1]
        return JaggedKeyValueFrame(arrs, new_index)

    def remove_values_smaller_than(self, value):
        arrs = OrderedDict((k, self[k].arr.remove_values_smaller_than(value)) for k in self.arrs)
        return JaggedKeyValueFrame(arrs, self.index)

    def cumsum(self):
        arrs = OrderedDict((k, self[k].cumsum().arr) for k in self.arrs)
        return JaggedKeyValueFrame(arrs, self.index)

    def get_ohlcv_frame(self, freq):
        dfs = OrderedDict((k, self[k].get_ohlcv_frame(freq)) for k in self.arrs)
        return pd.concat(dfs, axis=1)

    def get_fixed_depth_frame(self, depth, reverse):
        dfs = OrderedDict((k, self[k].get_fixed_depth_frame(depth, reverse)) for k in self.arrs)
        return pd.concat(dfs, axis=1)