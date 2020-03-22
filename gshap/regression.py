"""Regression explanations


"""


class RegressionDistance():
    def __init__(self, positive_density=None, negative_density=None):
        self.positive_density = positive_density
        self.negative_density = negative_density

    def __call__(self, output):
        p_pos = self._compute_probability(self.positive_density, output)
        if self.negative_density:
            p_neg = self._compute_probability(self.negative_density, output)
        else:
            p_neg = 1 - p_pos
        return 1 / (1 + (p_neg/p_pos).prod())

    def _compute_probability(self, funcs, output):
        if funcs is None:
            funcs = []
        elif not isinstance(funcs, list):
            funcs = [funcs]
        return sum([func(output) for func in funcs])