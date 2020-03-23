"""Regression explanations

The RegressionDistance class measures how much more likely each predicted 
target value (output) was to have been generated by a 'positive' density, 
rather than a 'negative' density.
"""


class RegressionDistance():
    """Computes regression distance

    Parameters
    ----------
    positive_density : callable or list of callables
        Densities take a (# samples x 1) vector of model outputs and return a 
        (# samples x 1) vector of probabilities that the output was generated 
        by the density.
    negative_densities : callable or list of callables
    """
    def __init__(self, positive_density=None, negative_density=None):
        self.positive_density = positive_density
        self.negative_density = negative_density

    def __call__(self, output):
        """Compute probability that the output follows a positive density"""
        p_pos = self._compute_probability(self.positive_density, output)
        if self.negative_density:
            p_neg = self._compute_probability(self.negative_density, output)
        else:
            p_neg = 1 - p_pos
        return 1 / (1 + (p_neg/p_pos).prod())

    def _compute_probability(self, funcs, output):
        """
        Compute the probability that each value of the output was generated by 
        one of the density functions.

        Parameters
        ----------
        funcs : None, callable, or list of callables
            Density functions
        output : np.array-like
            (# samples x 1) model output vector
        """
        if funcs is None:
            funcs = []
        elif not isinstance(funcs, list):
            funcs = [funcs]
        return sum([func(output) for func in funcs])