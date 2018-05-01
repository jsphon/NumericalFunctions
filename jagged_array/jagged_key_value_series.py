'''
Created on 1 Aug 2015

@author: jon
'''

import datetime
from collections import defaultdict

import numba as nb
import pandas as pd
import numpy as np

from jagged_array.jagged_key_value_array import JaggedKeyValueArray, INT_TYPES


# from numerical_functions import numba_funcs as nf

# INT_TYPES = (int, np.int, np.int64)
# DEFAULT_DTYPE = np.float32


class JaggedKeyValueSeries(object):
    """
    Object containing and indexed JaggedKeyValueArray.
    The purpose of using the index is to allow resampling.

    For example, if the underlying JaggedKeyValueArray was sampled unevenly,
    roughly every 0.5 seconds, we might like to resample it to 1s or 5s intervals.
    """

    def __init__(self, arr=None, index=None, keys=None, values=None, bounds=None):

        if index is None:
            index = np.arange(len(arr) - 1)
        elif isinstance(index, (list, tuple)):
            index = np.array(index)

        if arr is None:
            arr = JaggedKeyValueArray(keys=keys, values=values, bounds=bounds)

        self._verify(arr, index)
        self.arr = arr
        self.index = index

    def _verify(self, arr, index):
        assert isinstance(arr, JaggedKeyValueArray)
        assert len(arr) == len(index), '%s!=%s'%(len(arr), len(index))
        assert is_sorted(index)

    def __bool__(self):
        return bool(self.arr.keys.shape[0])

    def __len__(self):
        return len(self.index)

    def __eq__(self, other):
        if not isinstance(other, JaggedKeyValueSeries):
            return False
        if len(self) != len(other):
            return False
        if not np.array_equal(self.index, other.index):
            return False
        if self.arr != other.arr:
            return False

        return True

    # def loc(self, i):
    #     """
    #     like __getitem__, but using the index
    #     :return:
    #     """
    #
    #     index0 = self.index.searchsorted(i)
    #     index1 = index0+1
    #
    #     i0 = self.bounds[index0]
    #     i1 = self.bounds[index1]
    #     return (self.keys[i0:i1], self.values[i0:i1])
    #
    # def loc_slice(self, first=None, last=None):
    #     """
    #     Like loc, with slicing
    #     :param first:
    #     :param last:
    #     :return: JaggedKeyValueArray
    #     """
    #     if first is not None:
    #         i0 = self.index.searchsorted(first)
    #     else:
    #         i0 = 0
    #
    #     if last is not None:
    #         i1 = self.index.searchsorted(last)
    #     else:
    #         i1 = len(self.bounds)-1
    #
    #     keys = self.keys[self.bounds[i0]:self.bounds[i1]]
    #     values = self.values[self.bounds[i0]:self.bounds[i1]]
    #
    #     return JaggedKeyValueArray(
    #         keys,
    #         values,
    #         self.bounds[i0:i1+1] - self.bounds[i0],
    #         index=self.index[i0:i1]
    #     )

    def __getitem__(self, i):

        if isinstance(i, INT_TYPES):
            ii = self.index.searchsorted(i)
            # print('i1 is %s' % ii)
            return self.arr[ii]

        if isinstance(i, slice):
            ii0 = self.index.searchsorted(i.start, side='left')
            if i.stop is None:
                ii1 = len(self.index)
            else:
                ii1 = self.index.searchsorted(i.stop, side='left')

            new_index = self.index[ii0:ii1]
            new_array = self.arr[ii0:ii1]
            return JaggedKeyValueSeries(new_array, new_index)

        raise Exception('Not implemented for slice %s' % str(i))

    def __repr__(self):

        if len(self) > 6:
            rows0 = '\n'.join(['\t%s,%s,' % x for x in self[:3]])
            rows1 = '\n'.join(['\t%s,%s,' % x for x in self[-4:]])
            return '[\n%s\n\t...\n%s\n]' % (rows0, rows1)
        else:
            rows = '\n'.join(['\t%s,' % str(x) for x in self])
            return '[\n%s\n]' % rows

    def cumsum(self):
        new_array = self.arr.cumsum()
        return JaggedKeyValueSeries(new_array, self.index)

    def get_ohlcv_frame_by_interval(self, freq):
        resampled_index = get_resample_indices(self.index, freq)
        ohlc = self.get_ohlc_by_interval(freq)
        v = self.get_v_by_index(freq)
        df = pd.DataFrame(
            ohlc,
            index=resampled_index,
            columns=['o', 'h', 'l', 'c']
        )
        df['v'] = v
        return df

    # def get_ohlcv_frame_by_date_index(self, freq):
    #     resampled_index = get_resampled_datetime_index(self.date_index, freq)
    #     ohlc = self.get_ohlc_by_date_index(freq)
    #     v = self.get_v_by_date_index(freq)
    #     df = pd.DataFrame(
    #         ohlc,
    #         index=resampled_index,
    #         columns=['o', 'h', 'l', 'c']
    #     )
    #     df['v'] = v
    #     return df

    def get_ohlc_by_interval(self, freq):
        """
        Convert this array into Open/High/Low/Close bars
        :param freq:
        :return:
        """

        resampled_bounds = self.get_resample_index_bounds(freq)
        return self._get_ohlc(resampled_bounds)

    def get_ohlc_by_date_index(self, freq):
        """
        Convert this array into Open/High/Low/Close bars
        :param freq:
        :return:
        """

        resampled_bounds = self.get_resample_date_index_bounds(freq)
        return self._get_ohlc(resampled_bounds)

    def _get_ohlc(self, resampled_bounds):

        # Needs to be a float so that we can have nans
        result = np.empty((len(resampled_bounds), 4), dtype=np.float)
        result[:] = np.nan

        for i in range(0, len(resampled_bounds)):

            open0 = resampled_bounds[i, 0]
            open1 = resampled_bounds[i, 1]
            close0 = resampled_bounds[i, 2]
            close1 = resampled_bounds[i, 3]

            # High and Low
            all_values = self.keys[open0:close1]
            if close1 > open0:
                result[i, 1] = all_values.max()
                result[i, 2] = all_values.min()
            else:
                result[i, 1] = np.nan
                result[i, 2] = np.nan

            # Open
            opening_values = self.keys[open0:open1]
            result[i, 0] = modified_median(opening_values)

            # Close
            closing_values = self.keys[close0:close1]
            result[i, 3] = modified_median(closing_values)

        return result

    def get_active_keys(self):
        """
        Return the keys that are used by a jagged array

        This is required, as if an array is a view of another, then some
        of the keys could be unused
        """
        all_keys = self.keys[self.bounds[0]:self.bounds[-1]]
        return np.unique(all_keys)

    def get_resample_index_bounds(self, interval):

        floored = self.index // interval
        return self._get_resample_bounds(floored)

    def get_resample_date_index_bounds(self, freq):
        """
        Return a matrix of indices used for resampling


        columns represent (open0, open1, close0, close1)
        :param date_range:
        :param freq:
        :return:
        """

        floored = self.date_index.floor(freq)
        return self._get_resample_bounds(floored)

    def _get_resample_bounds(self, floored_index):

        i1 = np.where(np.diff(floored_index))[0] + 1
        i0 = np.array([0])
        open_bound_start_indices = np.r_[i0, i1]
        open_bound_end_indices = open_bound_start_indices + 1

        close_bound_start_indices = open_bound_start_indices[1:] - 1
        close_bound_end_indices = close_bound_start_indices + 1

        result = np.empty((len(open_bound_start_indices), 4), dtype=np.int)
        result[:, 0] = self.bounds[open_bound_start_indices]
        result[:, 1] = self.bounds[open_bound_end_indices]
        result[:-1, 2] = self.bounds[close_bound_start_indices]
        result[-1, 2] = self.bounds[-2]
        result[:-1, 3] = self.bounds[close_bound_end_indices]
        result[-1, 3] = self.bounds[-1]

        return result

    def get_v_by_date_index(self, freq):
        indices = get_resampled_datetime_index(self.date_index, freq)
        return self._get_resampled_v(indices)

    def get_v_by_index(self, freq):

        indices = get_resample_indices(self.index, freq)
        return self._get_resampled_v(indices)

    def _get_resampled_v(self, indices):

        result = np.ndarray(len(indices), dtype=self.values.dtype)

        extended_indices = np.append(indices, len(self.bounds))
        extended_bounds = np.append(self.bounds, len(self.values))
        for i in range(0, len(indices)):
            open_index0 = extended_bounds[extended_indices[i]]
            closing_index1 = extended_bounds[extended_indices[i + 1]]

            # High and Low
            all_values = self.values[open_index0:closing_index1]
            result[i] = all_values.sum()

        return result

    def ravel(self):
        """
        Return a representation consisting of 3 arrays
         - index
         - keys
         - values
        :return:
        """

        l = len(self)
        length = self.bounds[-1] - self.bounds[0]
        index = np.ndarray(length, dtype=self.index.dtype)
        lower_bound = self.bounds[0]
        for i in range(l):
            b0 = self.bounds[i]
            b1 = self.bounds[i + 1]
            i0 = b0 - lower_bound
            i1 = b1 - lower_bound
            index[i0:i1] = self.index[i]

        values = self.values[self.bounds[0]:self.bounds[-1]]
        keys = self.keys[self.bounds[0]:self.bounds[-1]]

        return index, keys, values

    def resample(self, freq):
        """
        Resample this, bucketting into data points floored to integer multiples
        of freq
        :param freq:
        :return:
        """
        floored = self.index // freq
        indices = get_change_indices(floored)

        cs = self.cumsum()

        data, unique_keys = cs.to_dense()

        old_row = np.zeros_like(unique_keys, dtype=np.int)

        row_diffs = []
        for i in indices[1:] - 1:
            row = data[i]
            row_diff = row - old_row
            row_diffs.append(row_diff)
            old_row = row

        last_row = data[-1]
        last_row_diff = last_row - old_row
        row_diffs.append(last_row_diff)

        diff_data = np.r_[row_diffs]
        result = JaggedKeyValueArray.from_dense(diff_data, unique_keys, dtype=np.int)
        result.index = floored[indices]
        return result

    @staticmethod
    def load(filename):

        with np.load(filename) as data:
            pdata = data['keys']
            sdata = data['values']
            bdata = data['bounds']
            index = data['index']

            return JaggedKeyValueSeries(pdata, sdata, bdata, index)

    def save(self, filename):
        """
        Save to a file
        :return:
        """

        data = {}
        data['keys'] = self.keys
        data['values'] = self.values
        data['bounds'] = self.bounds
        data['index'] = self.index

        np.savez_compressed(filename, **data)


def modified_median(x):
    """
    Return the median if x has an odd number of values
    if x has an odd number of values, return the lower of median values
    :param x list: sorted list of objects
    :return: median
    """
    if len(x):
        return x[(-1 + x.shape[0]) // 2]
    else:
        return np.nan


# def get_resampled_datetime_index(date_range, freq):
#     """
#     Return a date_range, resampled
#     :param date_range:
#     :param freq:
#     :return:
#     """
#     floored = date_range.floor(freq)
#     changed_indices = get_change_indices(floored)
#     return date_range[changed_indices]


def get_resampled_index(index, interval):
    """
    Resample the index, returning an array
    :param index: np.ndarray
    :param interval: int
    :return: np.ndarray
    """
    resampled_indices = get_resample_indices(index, interval)
    return index[resampled_indices]


def get_resample_indices(index, interval):
    """
    Return the integer indices representing the 1st index of each resampled
    bin. i.e. the index that would represent the open of an ohlc bar.
    :param index:
    :param freq:
    :return:
    """
    # TODO: Might be inefficient, as get_resampled_indices could be called
    # in the parent
    floored = floor_to_nearest_int(index, interval)
    return get_change_indices(floored)


@nb.jit(nopython=True)
def floor_to_nearest_int(x, multiplier):
    """
    Floor each value of x to the nearest multiple of multiplier
    :param x: array
    :param multiplier:
    :return:
    """
    return multiplier * (x // multiplier)


def get_change_indices(x):
    i1 = np.where(np.diff(x))[0] + 1
    i0 = np.array([0])
    return np.r_[i0, i1]


@nb.jit(nopython=True)
def is_sorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True
