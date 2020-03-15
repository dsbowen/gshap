"""KernelExplainer class

A model-agnostic class for computing general SHAP (G-SHAP) values.
"""

import datasets

from random import choices, shuffle
import numpy as np
import pandas as pd


class KernelExplainer():
    """Implements the Kernel SHAP method

    Parameters
    ----------
    model : callable
        Callable which takes a (# samples x # features) matrix and returns an 
        output which will be fed into `g`. For ordinary SHAP, the model 
        returns a (# samples x 1) output vector.
    
    data : numpy.array or pandas.DataFrame
        Background dataset from which values are randomly sampled to simulate 
        absent features.

    g : callable
        Callable which takes the `model` output and returns a scalar.
    """
    def __init__(self, model, data, g=(lambda x: x.mean())):
        self.model = model
        self.data = (
            data.values if isinstance(data, (pd.DataFrame, pd.Series)) 
            else data
        )
        self.N, self.P = data.shape
        self.g = g

    @property
    def nsamples(self):
        """Default number of samples to draw to approximate G-SHAP values"""
        return 2 * self.P + 2**11
        
    def gshap_values(self, X, **kwargs):
        """Compute G-SHAP values for all features

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            A (# samples x # features) matrix.
        nsamples : scalar or 'auto' (optional)
            Number of samples to draw when approximating G-SHAP values.

        Returns
        -------
        List of G-SHAP values ordered by feature index.
        """
        return np.array(
            [self.gshap_value(j, X, **kwargs) for j in range(self.P)]
        )

    def gshap_value(self, j, X, **kwargs):
        """Compute G-SHAP value for feature `j`

        Parameters
        ----------
        j : scalar or column name
            The index or column name of the feature of interest.
        X : numpy.array or pandas.DataFrame
            A (# samples x # features) matrix.
        nsamples : scalar or 'auto' (optional)
            Number of samples to draw when approximating G-SHAP values.

        Returns
        -------
        Approximated G-SHAP value for feature `j` (float).
        """
        j = X.columns.index(j) if isinstance(j, str) else j
        nsamples = kwargs.get('nsamples', self.nsamples)
        phi = [self._compute_phi(j, X) for m in range(nsamples)]
        return sum(phi) / len(phi)

    def _compute_phi(self, j, X):
        """Approximate G-SHAP value for feature `j` for one sample
        
        This method approximates the G-SHAP value by Monte Carlo sampling.
        1. Construct `Z` by sampling observations from the background dataset.
        2. Shuffle the order of the features.
        3. Construct `X_mj` (X minus the j'th feature) as all features from X 
        which come before j. Absent features are filled in from `Z`.
        4. Construct `X_pj` (X plus the j'th feature) by adding the original 
        j'th feature from `X` to `X_mj`.
        5. Return phi = g(model(X_pj)) - g(model(X_mj)).
        """
        X = X.values if isinstance(X, (pd.Series, pd.DataFrame)) else X
        if len(X.shape) == 1:
            # Reshape X to a 1xP matrix
            X = X.reshape(1, X.shape[0])
        # Ensure feature dimension of X matches that of the background data
        assert X.shape[1] == self.P

        Z = np.array(choices(self.data, k=X.shape[0]))

        order = np.array(list(range(self.P)))
        np.random.shuffle(order)
        j_idx = order[j]

        X_mj = (
            (order < j_idx).astype(int) * X 
            + (order >= j_idx).astype(int) * Z
        )
        X_pj = X_mj.copy()
        X_pj[:,j] = X[:,j]

        return self.g(self.model(X_pj)) - self.g(self.model(X_mj))