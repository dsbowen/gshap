"""Example datasets"""

import pandas as pd

import os

file_dir = os.path.dirname(os.path.abspath(__file__))

def load_recidivism():
    path = os.path.join(file_dir, 'compas/two-year-recidivism.csv')
    df = pd.read_csv(path)
    return df.drop('two_year_recid', axis=1), df['two_year_recid']