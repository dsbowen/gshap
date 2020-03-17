"""Intergroup difference measures and BlindClassfiier"""

import pandas as pd
import numpy as np
import scipy

from random import choices

def convert_to_np(vec):
    # Convert vector to boolean values
    if isinstance(vec, (pd.DataFrame, pd.Series)):
        return vec.values.astype(bool)
    return vec

def convert_proba(vec):
    # Convert probability output from a predict_proba method to probability 
    # of being in the positive class
    if len(vec.shape) == 1:
        return vec
    return vec[:,1]

def mean_absolute_distance(out_0, out_1):
    # Absolute difference between group means
    out_0, out_1 = [convert_proba(vec) for vec in (out_0, out_1)]
    return out_1.mean() - out_0.mean()

def mean_relative_distance(out_0, out_1):
    # Relative difference between group means
    out_0, out_1 = [convert_proba(vec) for vec in (out_0, out_1)]
    return out_1.mean() / out_0.mean() - 1


class IntergroupDifference():
    """Measures intergroup differences

    Paramters
    ---------
    group : numpy.array or pandas.Series
        (# samples x 1) array of boolean or binary values indicating 
        membership in the sensitive class.

    distance : callable
        Takes two vectors rep.
    """
    def __init__(self, group, distance=mean_absolute_distance):
        group = convert_to_np(group)
        self.group = group.astype(bool)
        self.distance = distance

    def __call__(self, output):
        """Compute distance measure between groups

        Parameters
        ----------
        ouput : numpy.array or pandas.Series
            (# samples x 1) binary vector

        Returns
        -------
        Scalar discrimination measure

        p0 is P(output=1|group=0); p1 is P(output=1|group=1)
        """
        out_0 = output[np.logical_not(self.group)]
        out_1 = output[self.group]
        return self.distance(out_0, out_1)


class BlindClassifier():
    """Blind a classifier to user-specified columns

    The blind classifier substitutes random values from the background 
    dataset into the blinded columns before sending the data to the 
    original classifier for prediction. It repeats this `nsamples` times, 
    then outputs the modal prediction.

    Parameters
    ----------
    model : callable
        Callable which takes a (# samples x # features) matrix and returns 
        a (# samples x 1) output vector.

    data : numpy.array, pandas.DataFrame, or pandas.Series
        Background dataset from which variables are randomly sampled to 
        blind the `model` to user-specified columns.

    columns : list of column names or indicies
        Columns to which the classifier is blinded.

    nsamples : scalar
        Number of random values sampled to produce the output.
    """
    def __init__(self, model, data, columns, nsamples=1):
        self.model = model
        if isinstance(data, pd.Series):
            self._from_pandas(data, list(data.index), columns)
        elif isinstance(data, pd.DataFrame):
            self._from_pandas(data, list(data.columns), columns)
        else:
            self.data = data
            self.columns = columns
        self.data = self._reshape_to_2d(self.data)
        # Boolean mask for blinded columns
        self.blinding_mask = np.zeros(self.data.shape[1])
        self.blinding_mask[self.columns] = 1
        # Boolean mask for visible columns
        self.visible_mask = np.logical_not(self.blinding_mask).astype(int)
        self.nsamples = nsamples

    def _from_pandas(self, data, columns, blind_columns):
        # Get data and columns from pandas DataFrame or Series
        self.data = data.values
        self.columns = [columns.index(c) for c in blind_columns]

    def _reshape_to_2d(self, X):
        # Reshape a (px1) vector to a (1xp) rector
        return X.reshape(1, X.shape[0]) if len(X.shape) == 1 else X

    def predict(self, X):
        # Classification prediction
        X = X.values if isinstance(X, (pd.Series, pd.DataFrame)) else X
        X = self._reshape_to_2d(X)
        assert X.shape[1] == self.data.shape[1]
        output = np.array(
            [self._compute_output(X) for i in range(self.nsamples)]
        )
        return scipy.stats.mode(output)[0][0]

    def _compute_output(self, X):
        # Substitute random values from the background dataset into the 
        # blinded coluns
        Z = np.array(choices(self.data, k=X.shape[0]))
        X_blind = self.visible_mask * X + self.blinding_mask * Z
        return self.model(X_blind)