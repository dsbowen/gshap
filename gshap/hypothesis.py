"""Hypothesis testing explanations

The HypothesisTest class measures how likely a hypothesis is to be true of 
an output vector. It uses a bootstrap analysis to compute a sample 
statistic, test a hypothesis using the sample statistic, and return the 
probability that the hypothesis test is passed.
"""

import numpy as np

from random import choices


class HypothesisTest():
    """Tests a hypothesis

    Parameters
    ----------
    statistic : callable
        Takes the model output and computes a sample statistic.
    test : callable
        Takes the sample statistic computed by `statistic` and returns the 
        result of the hypothesis test.
    bootstrap_samples : scalar
        Number of bootstrap samples for hypothesis testing.
    """
    def __init__(self, statistic, test, bootstrap_samples=1000):
        self.statistic = statistic
        self.test = test
        self.bootstrap_samples = bootstrap_samples

    def __call__(self, output):
        """Computes the probablity of the hypothesis being true"""
        test_results = [
            self._test_hypothesis(output) 
            for i in range(self.bootstrap_samples)
        ]
        return sum(test_results) / self.bootstrap_samples

    def _test_hypothesis(self, output):
        """Tests the hypothesis for one bootstrapped sample"""
        sample = np.array(choices(output, k=output.shape[0]))
        return float(self.test(self.statistic(sample)))