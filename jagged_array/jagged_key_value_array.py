'''
Created on 1 Aug 2015

@author: jon
'''


import datetime
from collections import defaultdict

import numba as nb
import pandas as pd
import numpy as np

from jagged_array.jagged_array import JaggedArray
from numerical_functions import numba_funcs as nf

INT_TYPES = (int, np.int, np.int64)
DEFAULT_DTYPE = np.float32


class JaggedKeyValueArray(object):

    def __init__(self, keys, values, bounds, dtype=None, index=None):

        dtype = dtype or DEFAULT_DTYPE

        if isinstance(keys, np.ndarray):
            self.keys = keys
        else:
            self.keys = np.array(keys, dtype=dtype)

        if isinstance(values, np.ndarray):
            self.values = values
        else:
            self.values = np.array(values, dtype=dtype)

        if isinstance(bounds, np.ndarray):
            self.bounds = bounds
        else:
            self.bounds = np.array(bounds, dtype=np.int)

        self.index = index

    @staticmethod
    def from_lists(key_list, val_list, dtype=None):
        """ Make a JaggedKeyValueArray from key and value list of lists
        """
        assert len(key_list) == len(val_list)

        bounds = np.ndarray(len(key_list) + 1, dtype=np.int)
        bounds[0] = 0
        c = 0
        for i, x in enumerate(key_list):
            c += len(x)
            bounds[i + 1] = c

        key_list = [x for item in key_list for x in item]
        val_list = [x for item in val_list for x in item]
        return JaggedKeyValueArray(key_list, val_list, bounds, dtype=dtype)

    @staticmethod
    def from_dense_nb(data, cols):  # , dtype=np.int ):
        """ Make a JaggedKeyValueArray from a dense array """
        keys, values, bounds = _from_dense_nb(data, cols)  # , dtype)
        n = bounds[-1]
        return JaggedKeyValueArray(keys[:n], values[:n], bounds)

    @staticmethod
    def from_dense(data, cols, dtype=np.int):
        """ Make a JaggedKeyValueArray from a dense array """

        keys = []
        values = []
        bounds = []
        for i in range(len(data)):
            row = data[i]
            mask = (row != 0)
            row_vals = row.compress(mask)
            row_cols = cols.compress(mask)

            keys.append(row_cols)
            values.append(row_vals)
            bounds.append(len(row_cols))

        return JaggedKeyValueArray.from_lists(keys, values)

    def get_histogram(self):
        result = defaultdict(int)
        for i in range(self.bounds[0], self.bounds[-1]):
            result[self.keys[i]] += self.values[i]
        return result

    def __bool__(self):
        return bool(self.keys.shape[0])

    def __len__(self):
        return self.bounds.shape[0] - 1

    def __eq__(self, other):
        if not isinstance(other, JaggedKeyValueArray):
            return False
        if len(self) != len(other):
            return False

        if not np.array_equal(self.bounds, other.bounds):
            return False
        if not np.array_equal(self.keys, other.keys):
            return False
        if not np.array_equal(self.values, other.values):
            return False

        if (self.index is not None) or (other.index is not None):

            return self.index==other.index

        return True

    def get_between(self, d0, d1):
        ''' Get a JaggedKeyValue array that is between d0 and d1.
        The result does not include d0 or d1
        It assumes the index is sorted.
        '''
        i0 = self.index.searchsorted(d0)+1
        i1 = self.index.searchsorted(d1)
        result = self[i0:i1]
        return result

    def __getitem__(self, i):

        if isinstance(self.index, pd.DatetimeIndex):
            if is_date_type(i):
                i0 = self.index.get_loc(i)
                return self[i0]

            if isinstance(i, slice)\
                    and (is_date_type(i.start) or is_date_type(i.stop)):

                i0 = self.index.get_loc(i.start) if i.start else None
                i1 = self.index.get_loc(i.stop) + 1 if i.stop else None

                s = slice(i0, i1, i.step)
                return JaggedKeyValueArray(
                    self.keys,
                    self.values,
                    self.bounds[s],
                    index=self.index[i0:i1]
                )

        if isinstance(i, INT_TYPES):
            i0 = self.bounds[i]
            i1 = self.bounds[i + 1]
            return (self.keys[i0:i1], self.values[i0:i1])

        if isinstance(i, slice):
            s = slice(i.start, i.stop + 1 if i.stop else None, i.step)
            if self.index is not None:
                s1 = slice(i.start, i.stop if i.stop else None, i.step)
                index = self.index[s1]
            else:
                index = None
            return JaggedKeyValueArray(self.keys, self.values, self.bounds[s], index=index)

        if isinstance(i, tuple):
            i0 = i[0]
            i1 = i[1]
            if isinstance(i0, INT_TYPES):
                if isinstance(i1, INT_TYPES) or isinstance(i1, slice):
                    j0 = self.bounds[i0]
                    j1 = self.bounds[i0 + 1]
                    return (self.keys[j0:j1][i1], self.values[j0:j1][i1])

            if isinstance(i0, slice):
                row_start = i0.start or 0
                row_end = i0.stop or len(self)
                if isinstance(i1, slice):
                    # Return fixed size arrays for the keys and values
                    assert i1.step is None, 'Only step 1 supported, step is %s' % i1.step
                    assert (i1.start is None) or (i1.stop is None)

                    if i1.start:
                        start = i1.start
                        assert start < 0
                        size = -start
                        key_result = np.zeros((len(self), size))
                        val_result = np.zeros((len(self), size))

                        ii = 0
                        for row in range(row_start, row_end + 1):
                            row_key, row_val = self[row]
                            rk = row_key[start:]
                            key_result[ii, start:] = rk
                            if rk.shape[0] < size:
                                key_result[ii, :size - rk.shape[0]] = 0
                            rv = row_val[start:]
                            val_result[ii, start:] = rv
                            if rv.shape[0] < size:
                                val_result[ii, :size - rv.shape[0]] = 0

                            ii += 1

                        return key_result, val_result

                elif isinstance(i1, INT_TYPES):
                    key_result = np.zeros(len(self))
                    val_result = np.zeros(len(self))

                    for i, row in enumerate(range(row_start, row_end)):
                        row_key, row_val = self[row]
                        key_result[i] = row_key[i1]
                        val_result[i] = row_val[i1]
                    return key_result, val_result

        raise Exception('Not implemented for slice %s' % str(i))

    def __repr__(self):

        if len(self) > 6:
            rows0 = '\n'.join(['\t%s,%s,' % x for x in self[:3]])
            rows1 = '\n'.join(['\t%s,%s,' % x for x in self[-4:]])
            return '[\n%s\n\t...\n%s\n]' % (rows0, rows1)
        else:
            rows = '\n'.join(['\t%s,' % str(x) for x in self])
            return '[\n%s\n]' % rows

    def get_keys_array(self):
        """ Return a jagged array of the keys """
        return JaggedArray(self.keys, self.bounds)

    def get_values_array(self):
        """ Return a jagged array of values """
        return JaggedArray(self.values, self.bounds)

    def to_dense_projection(self, projection, default_value=0):
        ''' Convert to a dense array, using projection as the keys
        '''
        d, k = self.to_dense(default_value=default_value)
        return _kv_to_dense_projection(d, k, projection)

    def to_dense(self, default_value=0):

        i0 = self.bounds[0]
        i1 = self.bounds[-1]
        keys = self.keys[i0:i1]
        vals = self.values[i0:i1]
        bounds = self.bounds - i0
        unique_keys, inverse_data = np.unique(keys, return_inverse=True)

        data = _kv_to_dense(keys, vals, bounds, unique_keys, inverse_data, default_value)

        return data, unique_keys

    def cumsum(self):
        unique_keys, inverse_data = np.unique(self.keys, return_inverse=True)

        if self:
            cs_keys, cs_vals, cs_bounds = _cumsum(self.keys, self.values, self.bounds, unique_keys, inverse_data)
            return JaggedKeyValueArray(cs_keys, cs_vals, cs_bounds)
        else:
            return JaggedKeyValueArray([], [], [])

    def get_ohlcv_frame(self, freq):
        resampled_index = get_resampled_index(self.index, freq)
        ohlc = self.get_ohlc(freq)
        v = self.get_v(freq)
        df = pd.DataFrame(
            ohlc,
            index=resampled_index,
            columns=['o', 'h', 'l', 'c']
        )
        df['v'] = v
        return df

    def get_ohlc(self, freq):
        """
        Convert this array into Open/High/Low/Close bars
        :param freq:
        :return:
        """

        resampled_bounds = self.get_resample_index_bounds(freq)

        # Needs to be a float so that we can have nans
        result = np.empty((len(resampled_bounds), 4), dtype=np.float)
        result[:] = np.nan

        for i in range(0, len(resampled_bounds)):

            open0 = resampled_bounds[i, 0]
            open1 = resampled_bounds[i, 1]
            close0 = resampled_bounds[i,2]
            close1 = resampled_bounds[i, 3]

            # High and Low
            all_values = self.keys[open0:close1]
            if close1>open0:
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

    def get_resample_index_bounds(self, freq):
        """
        Return a matrix of indices used for resampling


        columns represent (open0, open1, close0, close1)
        :param date_range:
        :param freq:
        :return:
        """

        floored = self.index.floor(freq)
        i1 = np.where(np.diff(floored))[0] + 1
        i0 = np.array([0])
        open_bound_start_indices = np.r_[i0, i1]
        open_bound_end_indices = open_bound_start_indices+1

        close_bound_start_indices = open_bound_start_indices[1:]-1
        close_bound_end_indices = close_bound_start_indices + 1

        result = np.empty((len(open_bound_start_indices), 4), dtype=np.int)
        result[:, 0] = self.bounds[open_bound_start_indices]
        result[:, 1] = self.bounds[open_bound_end_indices]
        result[:-1, 2] = self.bounds[close_bound_start_indices]
        result[-1, 2] = self.bounds[-2]
        result[:-1, 3] = self.bounds[close_bound_end_indices]
        result[-1, 3] = self.bounds[-1]

        return result

    def get_v(self, freq):

        indices = get_resample_indices(self.index, freq)

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

    # def get_hl(self, freq):
    #     resampled = self.resample(freq)
    #     result = []
    #     for row in resampled.get_keys_array():
    #         result.append([np.max(row), np.min(row)])
    #     return np.array(result)
    #
    # def get_h(self, freq):
    #     resampled = self.resample(freq)
    #     result = []
    #     for row in resampled.get_keys_array():
    #         result.append(np.max(row))
    #     return np.array(result)
    #
    # def get_c(self, freq):
    #     indices = get_resample_indices(self.index, freq)
    #     keys = self.get_keys_array()
    #
    #     result = np.ndarray(len(indices)+1, dtype=keys.dtype)
    #
    #     for c, i in enumerate(indices):
    #         row = keys[i]
    #         median = row[(-1+row.shape[0])//2]
    #         result[c] = median
    #
    #     row = keys[len(keys)-1]
    #     result[-1] = row[(-1+row.shape[0])//2]
    #     return result
    #
    # def get_o(self, freq):
    #     indices = get_resample_indices(self.index, freq)
    #     keys = self.get_keys_array()
    #
    #     result = np.ndarray(len(indices)+1, dtype=keys.dtype)
    #
    #     row = keys[0]
    #     result[0] = row[(-1+row.shape[0])//2]
    #     for c, i in enumerate(indices):
    #         row = keys[i+1]
    #         median = row[(-1+row.shape[0])//2]
    #         result[c+1] = median
    #     return result

    def resample(self, freq):
        indices = get_resample_indices(self.index, freq)
        cs = self.cumsum()

        data, unique_keys = cs.to_dense()

        old_row = np.zeros_like(unique_keys, dtype=np.int)

        row_diffs = []
        for i in indices[1:]-1:
            row = data[i]
            row_diff = row-old_row
            row_diffs.append(row_diff)
            old_row = row

        last_row = data[-1]
        last_row_diff = last_row - old_row
        row_diffs.append(last_row_diff)

        diff_data = np.r_[row_diffs]
        result = JaggedKeyValueArray.from_dense(diff_data, unique_keys, dtype=np.int)

        floored = self.index.floor(freq)
        result.index = floored[indices]
        return result


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


def get_resampled_index(date_range, freq):
    """
    Return a date_range, resampled
    :param date_range:
    :param freq:
    :return:
    """
    floored = date_range.floor(freq)
    i1 = np.where(np.diff(floored))[0]+1
    i0 = np.array([0])

    indices = np.r_[i0, i1]

    return floored[indices]


def get_resample_indices(date_range, freq):
    """
    Return the integer indices representing the 1st index of each resampled
    bin. i.e. the index that would represent the open of an ohlc bar.
    :param date_range:
    :param freq:
    :return:
    """
    # TODO: Inefficient, date_range.floor also called in parent
    floored = date_range.floor(freq)
    i1 = np.where(np.diff(floored))[0] + 1
    i0 = np.array([0])
    return np.r_[i0, i1]


def is_date_type(x):
    return isinstance(x, (datetime.datetime, pd.Timestamp))


@nb.jit(nopython=True)
def _cumsum(keys, values, bounds, unique_keys, inverse_data):
    buffer = np.zeros_like(unique_keys, dtype=values.dtype)
    max_possible_length = (bounds.shape[0] - 1) * unique_keys.shape[0]
    cs_keys = np.empty(max_possible_length, dtype=keys.dtype)
    cs_vals = np.empty(max_possible_length, dtype=values.dtype)
    cs_bounds = np.empty_like(bounds)
    cs_bounds[0] = 0
    pos = 0
    for i in range(bounds.shape[0] - 1):
        i0 = bounds[i]
        i1 = bounds[i + 1]
        for j in range(i0, i1):
            bcol = inverse_data[j]
            buffer[bcol] += values[j]

        for j in range(unique_keys.shape[0]):
            if buffer[j]:
                cs_vals[pos] = buffer[j]
                cs_keys[pos] = unique_keys[j]
                pos += 1
        cs_bounds[i + 1] = pos

    return cs_keys[:pos].copy(), cs_vals[:pos].copy(), cs_bounds


@nb.jit(nopython=True)
def _kv_to_dense_projection(d, k, projection):
    r = np.zeros((d.shape[0], projection.shape[0]), d.dtype)
    for i in range(r.shape[1]):
        v = projection[i]
        j = nf.binary_search(k, v)
        if (j < k.shape[0]) and (k[j] == v):
            for row in range(d.shape[0]):
                r[row, i] = d[row, j]
    return r


@nb.jit(nopython=True)
def _kv_to_dense(key_data, val_data, bounds, unique_keys, inverse_data, default_value):
    data = np.zeros((len(bounds) - 1, len(unique_keys)), dtype=key_data.dtype)
    for i in range(bounds.shape[0] - 1):
        i0 = bounds[i]
        i1 = bounds[i + 1]

        for j in range(i1 - i0):
            col_idx = inverse_data[i0 + j]
            data[i, col_idx] = val_data[i0 + j]

    return data


@nb.jit(nopython=True)
def _from_dense_nb(data, cols):  # , dtype=np.int ):
    """ Make the params for from_dense_nb """
    l = data.size
    keys = np.empty(l, cols.dtype)
    values = np.empty(l, data.dtype)
    bounds = np.empty(data.shape[0] + 1, np.int_)
    bounds[0] = 0
    c = 0
    for i in range(data.shape[0]):
        row = data[i]
        for j in range(data.shape[1]):
            if row[j]:
                keys[c] = cols[j]
                values[c] = row[j]
                c += 1
        bounds[i + 1] = c
    return keys, values, bounds
