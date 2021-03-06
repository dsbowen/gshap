"""G-SHAP utilities"""

import pandas as pd

def get_columns(X):
    """Get columns

    Parameters
    ----------
    X : numpy.array or pandas.DataFrame or pandas.Series

    Returns
    -------
    columns : list or None
    """
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    if isinstance(X, pd.Series):
        return list(X.index)

def get_data(X):
    """Get data matrix

    Parameters
    ----------
    X : numpy.array or pandas.DataFrame or pandas.Series

    Returns
    -------
    (# samples x # features) numpy.array
    """
    X = X.values if isinstance(X, (pd.Series, pd.DataFrame)) else X
    return X.reshape(1, X.shape[0]) if len(X.shape)==1 else X