import pandas as pd


def get_num_cat_cols(df) -> tuple:
    """Sort out numerical and categorical columns in a DataFrame

    Arguments
    ---------
    df: pd.DataFrame
        The DataFrame whose columns you want to sort out.

    Returns
    -------
    num_cols: list
        List containing names of the numerical features.

    cat_cols: list
        List containing names of the categorical features.
    """
    assert isinstance(df, pd.DataFrame)
    cat_cols = [col for col, dtype in df.dtypes if dtype == object]
    num_cols = [col for col in df.columns if col not in cat_cols]
    return num_cols, cat_cols
