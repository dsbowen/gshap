"""#General classification and regression explanations

For examples and interpretation, see my notebooks on [general classification explanations](https://github.com/dsbowen/gshap/blob/master/classification.ipynb) and [general regression explanations](https://github.com/dsbowen/gshap/blob/master/regression.ipynb).
"""


class ProbabilityDistance():
    """
    This class measures how likely each predicted target value (output) was to 
    have been generated by a 'positive' distribution or density, rather than a 
    'negative' distribution or density.

    Parameters
    ----------
    positive : callable or list of callables
        Densities and distributions take the output of a model, usually a 
        (# observations,) or (# observations, # classes) vector. It returns a 
        (# observations,) vector of probabilities that the predicted target 
        value was generated by the density or distribution.
        
    negative : callable or list of callables or None, default=None
        Similarly defined. If `None`, the probability that each observation 
        comes from a negative density or distribution will be treated as the 
        complement of `positive`.

    Attributes
    ----------
    positive : callable or list of callables
        Set from the `positive` parameter.

    negative : callable or list of callables
        Set from the `negative` parameter.

    Examples
    --------
    ```python
    import gshap
    from gshap.probability_distance import ProbabilityDistance

    from sklearn.datasets import load_iris
    from sklearn.svm import SVC

    X, y = load_iris(return_X_y=True)
    clf = SVC(probability=True).fit(X,y)

    # probability that each observation is in class 1
    pos_distribution = lambda y_pred: y_pred[:,1]
    # probability that each observation is in class 0
    neg_distribution = lambda y_pred: y_pred[:,0]
    g = ProbabilityDistance(pos_distribution, neg_distribution)
    explainer = gshap.KernelExplainer(clf.predict_proba, X, g)
    explainer.gshap_values(x, nsamples=1000)
    ```

    Out:

    ```
    array([0.02175944, 0.01505252, 0.17106646, 0.13605429])
    ```
    """
    def __init__(self, positive, negative=None):
        self.positive = positive
        self.negative = negative

    def __call__(self, output):
        """
        Parameters
        ----------
        output : np.array
            Model output, usually (# observations,) or 
            (# obervations, # classes) array for regression or classification 
            problems, respectively.

        Returns
        -------
        probability : float
            Probability that every predicted target value was generated by 
            a positive density or distribution, rather than a negative density 
            or distribution.
        """
        p_pos = self._compute_probability(self.positive, output)
        if self.negative:
            p_neg = self._compute_probability(self.negative, output)
        else:
            p_neg = 1 - p_pos
        x = 1 / (1 + (p_neg/p_pos).prod())
        return x

    def _compute_probability(self, funcs, output):
        """
        Compute the probability that each value of the output was generated by 
        one of the density or distribution functions.

        Parameters
        ----------
        funcs : None, callable, or list of callables
            Density functions

        output : np.array-like
            (# observations,) model output vector
        """
        if funcs is None:
            funcs = []
        elif not isinstance(funcs, list):
            funcs = [funcs]
        return sum([func(output) for func in funcs])