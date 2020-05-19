"""WeightedR2 class

An accuracy metric to assess model failure.

In progress.
"""

from gshap.utils import get_data

import numpy as np
import pandas as pd

from random import choice, shuffle


class Performance():
    def __init__(self, score, model, data, y, nsamples=1000):
        self.score = score
        self.set_sample_weight(model, data, y, nsamples)

    def set_sample_weight(self, model, data, y, nsamples=1000):
        data = get_data(data)
        y = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y

        scores = []
        for i in range(nsamples):
            Z = np.array(choices(data), k=y.shape[0])
            scores.append(1/self.score(y, model(Z)))
        self.sample_weight = np.array(scores).mean(axis=0)