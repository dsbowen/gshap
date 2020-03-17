"""Example datasets"""

import pandas as pd

import os

file_dir = os.path.dirname(os.path.abspath(__file__))

def load_dataset(filename, target, return_X_y=False):
    bunch = Bunch(filename, target)
    if return_X_y:
        return bunch.data.values, bunch.target.values
    return bunch

def load_recidivism(return_X_y=False):
    """COMPAS recidivism dataset"""
    return load_dataset(
        'compas/two-year-recidivism.csv', 
        target='two_year_recid',
        return_X_y=return_X_y
    )

class Bunch():
    def __init__(self, filename, target):
        self.df = pd.read_csv(os.path.join(file_dir, filename))
        self.data = self.df.drop(target, axis=1)
        self.target = self.df[target]