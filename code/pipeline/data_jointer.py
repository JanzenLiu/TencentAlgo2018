import pandas as pd
import numpy as np


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


class PandasMatrixJointer:
    def __init__(self, on):
        self.on = on

    @staticmethod
    def get_join_indices(df, on, row_names):
        assert isinstance(row_names, (list, np.ndarray))
        index_dict = list_to_dict(row_names)  # dict
        return df[on].map(index_dict).values

    @staticmethod
    def quick_join(matrix, row_indices):
        return matrix[row_indices, :]

    @staticmethod
    def static_join(df, matrix, on, row_names):
        row_indices = PandasMatrixJointer.get_join_indices(df, on, row_names)
        return PandasMatrixJointer.quick_join(matrix, row_indices)

    def join(self, df, matrix, row_names):
        return PandasMatrixJointer.static_join(df, matrix, self.on, row_names)


class PandasPandasJointer:
    def __init__(self, on):
        self.on = on

    @staticmethod
    def quick_join(df1, df2, on):
        return pd.merge(df1, df2, on=on, how="left")

    def join(self, df1, df2):
        return PandasPandasJointer.quick_join(df1, df2, self.on)
