import scipy.sparse as sparse
import pandas as pd
import numpy as np


def dict_to_list(dic):
    """Given a dictionary mapping something to integers, return list of keys sorted by their values"""
    return [k for k, v in sorted(dic.items(), key=lambda x: x[1])]


def list_to_dict(lst, step=1):
    """Get Mapping from list elements to their indices in the list.

    Parameters
    ----------
    lst: list
        List containing some values.

    step: int
        Step of the index to generate.

    Returns
    --------
    val_dict: dict
        Dictionary that maps value in the given list to their indices in the list.
    """
    val_dict = dict(zip(lst, list(range(0, len(lst) * step, step))))
    return val_dict


def get_row_index(df, row_vals, row_for="uid"):
    val_to_index = list_to_dict(row_vals)
    row_indices = df[row_for].map(val_to_index)
    return row_indices.values


def sum_column(matrix):
    """Sum up each column of a given csr_matrix.

    Parameters
    ----------
    matrix: sparse.csr_matrix
        Matrix whose columns you are going to sum up. Shape (n_rows, n_cols)

    Returns
    -------
    arr_sum: np.ndarray
        Array containing column sums calculated from the given matrix. Shape (n_cols, )
    """
    arr_sum = matrix.sum(axis=0)
    arr_sum = np.asarray(arr_sum)  # convert matrix to array
    arr_sum = np.squeeze(arr_sum)  # convert 2-D array to 1-D array
    return arr_sum


class MatrixCounter:
    def __init__(self, matrix, col_names, row_names, df,
                 groupby="aid", row_for="uid", label="label", gvals=None, lvals=None):
        # please make sure there is no NaN in df before calling the constructor
        self.matrix = matrix
        self.col_names = col_names
        self.row_names = row_names
        self.groupby = groupby
        self.row_for = row_for
        self.label = label
        self.gvals = gvals if gvals is not None else df[groupby].unique().tolist()
        self.lvals = lvals if lvals is not None else df[label].unique().tolist()
        self.df = df.copy()  # to avoid from mutating the original DataFrame
        self.df["rowIndex"] = self.build_row_index()
        self.group_dict = self.build_index_dict()

    def build_row_index(self):
        return get_row_index(self.df, self.row_names, self.row_for)

    def build_index_dict(self):
        return dict(self.df.groupby([self.groupby, self.label])["rowIndex"].apply(set))

    def get_indices(self, gval, lval, tolist=True):
        indices = self.group_dict.get((gval, lval), set())
        indices = list(indices) if tolist else indices
        return indices

    def get_slice(self, gval, lval):
        indices = self.get_indices(gval, lval, tolist=True)
        return self.matrix[indices, :]

    def conditional_count(self, gval, lval):
        return sum_column(self.get_slice(gval, lval))

    def group_count(self, gval):
        df_records = pd.DataFrame.from_dict({lval: self.conditional_count(gval, lval) for lval in self.lvals})
        df_records["value"] = self.col_names
        return df_records

    def __del__(self):
        del self.df
        del self.group_dict


class MatrixCounterManager:
    def __init__(self, matrix, col_names, row_names, df,
                 groupby="aid", row_for="uid", label="label", gvals=None, lvals=None):
        # please make sure there is no NaN in df before calling the constructor
        self.matrix = matrix
        self.col_names = col_names
        self.row_names = row_names
        self.groupby = groupby
        self.row_for = row_for
        self.label = label
        self.gvals = gvals if gvals is not None else df[groupby].unique().tolist()
        self.lvals = lvals if lvals is not None else df[label].unique().tolist()
        self.df = df

    def build_matrix_counter(self, indices):
        df = self.df.iloc[indices]
        return MatrixCounter(self.matrix, self.col_names, self.row_names, df,
                             self.groupby, self.row_for, self.label, self.gvals, self.lvals)
