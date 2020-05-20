"""# Kernel Explainer"""

from gshap.utils import get_data

from random import choices, shuffle
import numpy as np
import pandas as pd


class KernelExplainer():
    """
    The Kernel Explainer is a model-agnostic method of approximating G-SHAP 
    values.

    Parameters
    ----------
    model : callable
        Callable which takes a (# observations, # features) matrix and returns
        an output which will be fed into `g`. For ordinary SHAP, the model 
        returns a (# observations, # targets) output vector.
    
    data : numpy.array or pandas.DataFrame or pandas.Series
        Background dataset from which values are randomly sampled to simulate 
        absent features.

    g : callable
        Callable which takes the `model` output and returns a scalar.

    Attributes
    ----------
    model : callable
        Set from the `model` parameter.

    data : numpy.array
        Set from the `data` parameter. If `data` is a `pandas` object, it is 
        automatically converted to a `numpy.array`.

    g : callable
        Set from the `g` parameter.

    Examples
    --------
    This example shows how to compute classical SHAP values.
    ```python
    import gshap
    
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression

    X, y = load_boston(return_X_y=True)
    reg = LinearRegression().fit(X,y)
    explainer = gshap.KernelExplainer(
    \    model=reg.predict, data=X, g=lambda x: x.mean()
    )
    explainer.gshap_values(X, nssamples=1000)
    ```

    Out:

    ```
    array([-8.52873964e-04, -4.90442234e-04,  9.42836482e-05,  3.98231297e-04,
    \    2.03149964e-03,  3.93086231e-03, -7.38176865e-06,  3.81400727e-03,
    \    5.19437337e-03, -1.34661588e-03,  7.08535145e-04,  1.50486721e-03,
    \   -8.28480438e-03])
    ```

    As expected, all SHAP values are 0 for linear regression. We can see this 
    when we compare the mean prediction for the original data `X` to the 
    shuffled background data `explainer.data`.

    ```python
    explainer.compare(X, bootstrap_samples=1000)
    ```

    Out:

    ```
    22.53280632411067, 22.52089950825812
    ```
    """
    def __init__(self, model, data, g=lambda x: x.mean()):
        self.model = model
        self.data = data
        self.g = g

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, X):
        self._data = (
            X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        )
        self.N, self.P = self._data.shape

    @property
    def nsamples(self):
        """Default number of samples to draw to approximate G-SHAP values"""
        return 2 * self.P + 2**11

    def compare(self, X, bootstrap_samples=1000):
        """
        Compares the background data `self.data` to the comparison data `X` 
        in terms of the general function `self.g`.

        Parameters
        ----------
        X : numpy.array or pandas.Series or pandas.DataFrame
            (# samples, # features) matrix of comparison data.

        bootstrap_samples : scalar
            Number of bootstrapped samples for computing `g` of the 
            background data.

        Returns
        -------
        g_comparison : float
            *g(model(X))*, where *X* is the comparison data.

        g_background : float
            *g(model(X_b))*, where *X_b* is the shuffled background data.
        """
        X = get_data(X)
        g_data = []
        for i in range(bootstrap_samples):
            sample = np.array(choices(self.data, k=X.shape[0]))
            g_data.append(self.g(self.model(sample)))
        return self.g(self.model(X)), sum(g_data) / len(g_data)
        
    def gshap_values(self, X, **kwargs):
        """
        Compute G-SHAP values for all features.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame or pandas.Series
            A (# samples, # features) matrix.

        nsamples : scalar or 'auto', default='auto'
            Number of samples to draw when approximating G-SHAP values.

        Returns
        -------
        gshap_values : np.array
            (# features,) vector of G-SHAP values ordered by feature index.
        """
        return np.array(
            [self.gshap_value(j, X, **kwargs) for j in range(self.P)]
        )

    def gshap_value(self, j, X, **kwargs):
        """
        Compute the G-SHAP value for feature `j`.

        Parameters
        ----------
        j : scalar or column name
            The index or column name of the feature of interest.

        X : numpy.array or pandas.DataFrame or pandas.Series
            A (# samples, # features) matrix.

        nsamples : scalar or 'auto', default='auto'
            Number of samples to draw when approximating G-SHAP values.

        Returns
        -------
        gshap_value : float
            Approximated G-SHAP value for feature `j` (float).
        """
        j = list(X.columns).index(j) if isinstance(j, str) else j
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
        X = get_data(X)
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