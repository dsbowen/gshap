<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<link rel="stylesheet" href="https://assets.readthedocs.org/static/css/readthedocs-doc-embed.css" type="text/css" />

<style>
    a.src-href {
        float: right;
    }
    p.attr {
        margin-top: 0.5em;
        margin-left: 1em;
    }
    p.func-header {
        background-color: gainsboro;
        border-radius: 0.1em;
        padding: 0.5em;
        padding-left: 1em;
    }
    table.field-table {
        border-radius: 0.1em
    }
</style>#Hypothesis testing

For examples and interpretation, see my notebook on [hypothesis test explanations](https://github.com/dsbowen/gshap/blob/master/hypothesis.ipynb).

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##gshap.hypothesis.**HypothesisTest**

<p class="func-header">
    <i>class</i> gshap.hypothesis.<b>HypothesisTest</b>(<i>test, bootstrap_samples=1000</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/hypothesis.py#L11">[source]</a>
</p>

This class measures how likely a hypothesis is to be true of an output
vector. It uses a bootstrap analysis to compute the probability that a
hypothesis is true of a population from a sample output vector.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>test : <i>callable</i></b>
<p class="attr">
    Takes an output vector and returns a boolean indicator that the hypothesis is true of the output vector. This will usually involve computing a sample statistic of the output vector, then returning an indicator that the sample statistic fell within a certain range.
</p>
<b>bootstrap_samples : <i>int, default=1000</i></b>
<p class="attr">
    Number of bootstrap samples for hypothesis testing.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>test : <i>callable</i></b>
<p class="attr">
    Set from the <code>test</code> parameter.
</p>
<b>bootstrap_samples : <i>int</i></b>
<p class="attr">
    Set from the <code>bootstrap_samples</code> parameter.
</p></td>
</tr>
    </tbody>
</table>

####Examples

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
    0.0009,  0.1415,  0.0071])
```

####Methods



<p class="func-header">
    <i></i> <b>__call__</b>(<i>self, output</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/hypothesis.py#L66">[source]</a>
</p>

Computes the probablity of the hypothesis being true of the population
from which the sample was drawn.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>output : <i>numpy.array</i></b>
<p class="attr">
    (# observations, # targets) vector of model outputs.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>probability : <i>scalar between 0 and 1</i></b>
<p class="attr">
    Probability that the hypothesis is true of the population from which the sample was drawn.
</p></td>
</tr>
    </tbody>
</table>

