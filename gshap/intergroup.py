"""Algorithm fairness measures

See https://arxiv.org/abs/2001.09784 for definitions.

Run on compas
Clean code
Write up
Can also use this for explain your move; g takes the output (vector of utilities assigned to moves) and maps it using the specificity and relevance criteria.
"""

import pandas as pd
import numpy as np

def convert_to_np(vec):
    return vec.values if isinstance(vec, (pd.DataFrame, pd.Series)) else vec


class DisparateImpact():
    """Measures disparate impact between groups 

    Paramters
    ---------
    group : numpy.array or pandas.Series
        (# samples x 1) array of boolean or binary values indicating 
        membership in the sensitive class.

    difference : 'relative' or 'absolute'
        Whether the difference is relative impact or absolute impact
    """
    def __init__(self, group, difference='relative'):
        group = convert_to_np(group)
        self.group = group.astype(bool)
        self.difference = difference

    @property
    def difference(self):
        return self._difference

    @difference.setter
    def difference(self, val):
        assert val in ('relative', 'absolute')
        self._difference = val

    def __call__(self, output):
        """Compute disparate impact measure

        Parameters
        ----------
        ouput : numpy.array or pandas.Series
            (# samples x 1) binary vector

        Returns
        -------
        Scalar discrimination measure

        p0 is P(output=1|group=0); p1 is P(output=1|group=1)
        """
        p0 = output[np.logical_not(self.group)].mean()
        p1 = output[self.group].mean()
        return p1/p0 - 1 if self.difference == 'relative' else p1-p0


class EqualOpportunity(DisparateImpact):
    """Measure of equal opportunity

    Similar to disparate impact, but only considers observations for which the 
    outcome is 1 or True.

    Parameters
    ----------
    y : np.array or pandas.Series
        (# samples x 1) boolean or binary outcome vector
    """
    def __init__(self, y, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.y = y.astype(bool)
        self.group = self.group[self.y]

    def __call__(self, output):
        return super().__call__(output[self.y])