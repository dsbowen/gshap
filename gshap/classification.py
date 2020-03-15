"""Classification distance measures

The ClassificationDistance class measures how much more likely observations 
are to be classified as one of the 'positive' classes rather than one of the 
'negative' classes.

Use this class to answer questions such as 'What distinguished a Versicolour 
iris from a Setosa?' and 'What distinguishes a Versicolour iris from a 
Virginica?'.

In these examples, the positive class is Versicolor, and the negative class 
is Setosa and Virginica, respectively.
"""

import numpy as np


class ClassificationDistance():
    """Computes classification distance G-SHAP values

    Parameters
    ----------
    positive classes : list
        A list of classes against which the output is compared for membership.
    negative_classes : list
        A list of classes against which the output is compared for 
        non-membership.
    """
    def __init__(self, positive_classes=None, negative_classes=None):
        assert positive_classes is not None or negative_classes is not None
        self.positive_classes = positive_classes
        self.negative_classes = negative_classes

    @property
    def classes(self):
        # Positive and negative classes, or 'all'
        if self.positive_classes is None or self.negative_classes is None:
            return 'all'
        return self.positive_classes + self.negative_classes

    def select_observations(self, model, *X):
        """Select observations belonging to the relevant classes

        Parameters
        ----------
        model : callable
            Takes features matrices X and returns (# samples x 1) 
            classification vector.
        X : (# samples x # features) matrices
        """
        if self.classes != 'all':
            X = [x[np.isin(model(x), self.classes)] for x in X]        
        return X[0] if len(X) == 1 else X

    def __call__(self, output):
        if len(output.shape) == 1:
            pos, neg = self._hard_classes(output)
        else:
            pos, neg = self._soft_classes(output)
        return pos / (pos + neg)

    def _hard_classes(self, output):
        """Compute positive and negative probabilities for hard classification

        `output` is a (# samples x 1) classification vector.
        """
        pos = np.isin(output, self.positive_classes)
        neg = np.isin(output, self.negative_classes, invert=True)
        return pos, neg

    def _soft_classes(self, output):
        """Compute positive and negative probabilities for soft classification

        `output` is a (# samples x # classes) matrix of probability of class 
        membership
        """
        pos = output[:,self.positive_classes].sum(axis=1).mean()
        neg = output[:,self.negative_classes].sum(axis=1).mean()
        return pos, neg

    # def _get_positive_classes(self, output):
    #     if self.positive_classes:
    #         return self.positive_classes
    #     classes = list(range(len(output)))
    #     return [c for c in classes if c not in self.negative_classes]

    # def _get_negative_classes(self, output):
    #     if self.negative_classes:
    #         return self.negative_classes
    #     classes = list(range(len(output)))
    #     return [c for c in classes if c not in self.positive_classes]