"""#Hypothesis testing

For examples and interpretation, see my notebook on [hypothesis test explanations](https://github.com/dsbowen/gshap/blob/master/hypothesis.ipynb).
"""

import numpy as np

from random import choices


class HypothesisTest():
    """
    This class measures how likely a hypothesis is to be true of an output 
    vector. It uses a bootstrap analysis to compute the probability that a 
    hypothesis is true of a population from a sample output vector.

    Parameters
    ----------
    test : callable
        Takes an output vector and returns a boolean indicator that the
        hypothesis is true of the output vector. This will usually involve
        computing a sample statistic of the output vector, then returning an
        indicator that the sample statistic fell within a certain range.

    bootstrap_samples : int
        Number of bootstrap samples for hypothesis testing.

    Attributes
    ----------
    test : callable
        Set from the `test` parameter.

    bootstrap_samples : int
        Set from the `bootstrap_samples` parameter.

    Examples
    --------
    ```python
    import gshap
    from gshap.hypothesis import HypothesisTest

    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import Lasso

    X, y = load_diabetes(return_X_y=True)
    reg = Lasso(alpha=.1).fit(X, y)

    test = lambda y_pred: y_pred.mean() > 155
    g = HypothesisTest(test, bootstrap_samples=100)
    explainer = gshap.KernelExplainer(reg.predict, X, g)
    # artifically select a sample which with higher-than-average y
    explainer.gshap_values(X[y > 70], nsamples=100)
    ```

    Out:

    ```
    array([-0.0069,  0.0253,  0.2572,  0.1112, -0.0108, -0.0105,  0.0317,
    \    0.0009,  0.1415,  0.0071])
    ```
    """
    def __init__(self, test, bootstrap_samples=1000):
        self.test = test
        self.bootstrap_samples = bootstrap_samples

    def __call__(self, output):
        """
        Computes the probablity of the hypothesis being true of the population
        from which the sample was drawn.
        
        Parameters
        ----------
        output : numpy.array
            (# observations, # targets) vector of model outputs.

        Returns
        -------
        probability : scalar between 0 and 1
            Probability that the hypothesis is true of the population from 
            which the sample was drawn.
        """
        test_results = [
            float(self.test(np.array(choices(output, k=output.shape[0]))))
            for i in range(self.bootstrap_samples)
        ]
        return sum(test_results) / self.bootstrap_samples