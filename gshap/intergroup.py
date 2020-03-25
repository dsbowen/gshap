"""Intergroup difference measures

The IntergroupDifference class measures the distance between distributions of 
predicted outcomes for different groups.
"""

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

def absolute_mean_distance(out_0, out_1):
    # Absolute difference between group means
    out_0, out_1 = [convert_proba(vec) for vec in (out_0, out_1)]
    return out_1.mean() - out_0.mean()

def relative_mean_distance(out_0, out_1):
    # Relative difference between group means
    out_0, out_1 = [convert_proba(vec) for vec in (out_0, out_1)]
    return out_1.mean() / out_0.mean() - 1


class IntergroupDifference():
    """Measures intergroup differences

    Paramters
    ---------
    group : numpy.array or pandas.Series
        (# samples x 1) array of boolean or binary values indicating 
        group membership.

    distance : callable
        Takes two vectors of model output for the outgroup and ingroup. 
        Output vectors will usually be (# outgroup x 1) and (# ingroup x 1), 
        or (# outgroup x # classes) and (# ingroup x # classes). Returns a 
        scalar measure of intergroup difference, such as the absolute 
        difference between group means.
    """
    def __init__(self, group, distance=absolute_mean_distance):
        group = convert_to_np(group)
        self.group = group.astype(bool)
        self.distance = distance

    def __call__(self, output):
        """Compute distance measure between groups

        Parameters
        ----------
        ouput : numpy.array or pandas.Series
            Model output, usually a (# samples x 1) or (# samples x # classes) 
            vector.

        Returns
        -------
        Scalar measure of distance between distribution of predicted output 
        for outgroup and ingroup.
        """
        out_0 = output[np.logical_not(self.group)]
        out_1 = output[self.group]
        return self.distance(out_0, out_1)