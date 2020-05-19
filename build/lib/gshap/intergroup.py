"""#Intergroup differences

For examples and interpretation, see my notebook on [intergroup difference explanations](https://github.com/dsbowen/gshap/blob/master/intergroup_difference.ipynb).
"""

import pandas as pd
import numpy as np
import scipy

from random import choices


class IntergroupDifference():
    """
    This class measures the distance between distributions of predicted 
    outcomes for different groups.

    Paramters
    ---------
    group : numpy.array or pandas.Series
        (# observations,) array of boolean or binary values indicating 
        group membership.

    distance : callable or str, default='absolute_mean_distance'
        Takes two vectors of model output for the outgroup and ingroup. 
        Output vectors will usually be (# outgroup,) and (# ingroup,), or 
        (# outgroup, # classes) and (# ingroup, # classes). `distance` returns
        a scalar measure of intergroup difference, such as the absolute 
        difference between group means. If input as a string, `distance` is
        used as a key to look up built-in distance functions.

    Attributes
    ----------
    group : numpy.array
        Set from the `group` parameter. If the parameter is passed as a 
        `pandas.Series`, it is automatically converted in a `numpy.array`.

    distance : callable or str
        Set from the `distance` parameter.

    Examples
    --------
    ```python
    import gshap
    from gshap.datasets import load_recidivism
    from gshap.intergroup import IntergroupDifference

    from sklearn.svm import SVC

    X, y = load_recidivism(return_X_y=True)
    clf = SVC().fit(X,y)

    g = IntergroupDifference(group=X_test['black'], distance='relative_mean_distance')
    explainer = gshap.KernelExplainer(clf.predict, X_train, g)
    explainer.gshap_values(X_test, nsamples=32)
    ```

    Out:

    ```
    array([ 0.01335252,  0.24884556,  0.00132373, -0.0025238 , -0.00151837,
    \    0.40453822,  0.01636782,  0.07666043, -0.00056414,  0.00966583])
    ```
    """
    def __init__(self, group, distance='absolute_mean_distance'):
        group = _convert_to_np(group)
        self.group = group.astype(bool)
        self.distance = (
            distance_metrics[distance] if isinstance(distance, str) else distance
        )

    def __call__(self, output):
        """
        Compute distance measure between groups.

        Parameters
        ----------
        ouput : numpy.array or pandas.Series
            Model output, usually a (# observations,) or 
            (# observations, # classes) vector.

        Returns
        -------
        distance : scalar
            Measure of the distance between the distributions of predicted 
            outputs for outgroup and ingroup observations.
        """
        out_0 = output[np.logical_not(self.group)]
        out_1 = output[self.group]
        return self.distance(out_0, out_1)


def absolute_mean_distance(out_0, out_1):
    """
    Parameters
    ----------
    out_0 : np.array
        (# observations,) vector of model outputs for outgroup observations.

    out_1 : np.array
        (# observations,) vector of model outputs for ingroup observations.

    Returns
    -------
    distance : scalar
        out_1.mean() - out_0.mean()
    """
    out_0, out_1 = [_convert_proba(vec) for vec in (out_0, out_1)]
    return out_1.mean() - out_0.mean()

def relative_mean_distance(out_0, out_1):
    """
    Parameters
    ----------
    out_0 : np.array
        (# observations,) vector of model outputs for outgroup observations.

    out_1 : np.array
        (# observations,) vector of model outputs for ingroup observations.

    Returns
    -------
    distance : scalar
        out_1.mean() / out_0.mean() - 1
    """
    out_0, out_1 = [_convert_proba(vec) for vec in (out_0, out_1)]
    return out_1.mean() / out_0.mean() - 1

def _convert_proba(vec):
    # Convert probability output from a predict_proba method to probability 
    # of being in the positive class
    if len(vec.shape) == 1:
        return vec
    return vec[:,1]

def _convert_to_np(vec):
    # Convert vector to boolean values
    if isinstance(vec, (pd.DataFrame, pd.Series)):
        return vec.values.astype(bool)
    return vec

distance_metrics = {
    'absolute_mean_distance': absolute_mean_distance,
    'relative_mean_distance': relative_mean_distance
}