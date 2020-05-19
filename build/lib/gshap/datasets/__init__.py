"""# Example datasets"""

import pandas as pd

import os

file_dir = os.path.dirname(os.path.abspath(__file__))

def _load_dataset(filename, target, return_X_y=False):
    """
    Load a dataset.

    Parameters
    ----------
    filename : str

    target : str
        Name of target variable

    return_X_y : bool
        Indicates to return just the X and y matrices.
    
    Returns
    -------
    bunch : Bunch
        Object containing the dataframe, X feature matrix, and y target 
        vector. Or, if `return_X_y`, return (X,y).
    """
    bunch = Bunch(filename, target)
    return (bunch.data.values, bunch.target.values) if return_X_y else bunch

def load_recidivism(return_X_y=False):
    """
    Load the COMPAS recidivism dataset. The purpose of this dataset is to 
    predict whether a criminal will recidivate within two years of release.
    
    Parameters
    ----------
    return_X_y : bool, default=False
        Indicates whether to return just the X and y matrices, as opposed 
        to the data `Bunch`.

    Returns
    -------
    bunch : Bunch
        Object containing the dataframe, X feature matrix, and y target 
        vector. Or, if `return_X_y`, return (X,y).
    """
    return load_dataset(
        'compas/two-year-recidivism.csv', 
        target='two_year_recid',
        return_X_y=return_X_y
    )

def load_gdp(return_X_y=False):
    """
    Load the GDP growth dataset (from FRED data). The purpose of this 
    dataset is to forecast GDP growth based on macroeconomic variables.
    
    Parameters
    ----------
    return_X_y : bool, default=False
        Indicates whether to return just the X and y matrices, as opposed 
        to the data `Bunch`.

    Returns
    -------
    bunch : Bunch
        Object containing the dataframe, X feature matrix, and y target 
        vector. Or, if `return_X_y`, return (X,y).
    """
    return load_dataset(
        'gdp/GDP-growth.csv',
        target='GDP_g',
        return_X_y=return_X_y
    )

class Bunch():
    """
    Dataset container.

    Parameters
    ----------
    filename : str
        Name of the file in `gshap/datasets`.

    target : str
        Name of target variable

    Attributes
    ----------
    df : pandas.DataFrame
        Dataframe containing features and the target variable.

    data : pandas.DataFrame
        Dataframe containing only the features

    target : pandas.Series
        Series of the target variable.
    """
    def __init__(self, filename, target):
        self.df = pd.read_csv(os.path.join(file_dir, filename))
        self.data = self.df.drop(target, axis=1)
        self.target = self.df[target]