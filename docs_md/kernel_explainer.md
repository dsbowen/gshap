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
</style># Kernel Explainer

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##gshap.**KernelExplainer**

<p class="func-header">
    <i>class</i> gshap.<b>KernelExplainer</b>(<i>model, data, g=lambda x: x.mean()</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/__init__.py#L10">[source]</a>
</p>

The Kernel Explainer is a model-agnostic method of approximating G-SHAP
values.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>model : <i>callable</i></b>
<p class="attr">
    Callable which takes a (# observations, # features) matrix and returns an output which will be fed into <code>g</code>. For ordinary SHAP, the model returns a (# observations, # targets) output vector.
</p>
<b>data : <i>numpy.array or pandas.DataFrame or pandas.Series</i></b>
<p class="attr">
    Background dataset from which values are randomly sampled to simulate absent features.
</p>
<b>g : <i>callable</i></b>
<p class="attr">
    Callable which takes the <code>model</code> output and returns a scalar.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>model : <i>callable</i></b>
<p class="attr">
    Set from the <code>model</code> parameter.
</p>
<b>data : <i>numpy.array</i></b>
<p class="attr">
    Set from the <code>data</code> parameter. If <code>data</code> is a <code>pandas</code> object, it is automatically converted to a <code>numpy.array</code>.
</p>
<b>g : <i>callable</i></b>
<p class="attr">
    Set from the <code>g</code> parameter.
</p></td>
</tr>
    </tbody>
</table>

####Examples

This example shows how to compute classical SHAP values.
```python
import gshap

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

X, y = load_boston(return_X_y=True)
reg = LinearRegression().fit(X,y)
explainer = gshap.KernelExplainer(
    model=reg.predict, data=X, g=lambda x: x.mean()
)
explainer.gshap_values(X, nssamples=1000)
```

Out:

```
array([-8.52873964e-04, -4.90442234e-04,  9.42836482e-05,  3.98231297e-04,
    2.03149964e-03,  3.93086231e-03, -7.38176865e-06,  3.81400727e-03,
    5.19437337e-03, -1.34661588e-03,  7.08535145e-04,  1.50486721e-03,
   -8.28480438e-03])
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

####Methods



<p class="func-header">
    <i></i> <b>compare</b>(<i>self, X, bootstrap_samples=1000</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/__init__.py#L102">[source]</a>
</p>

Compares the background data `self.data` to the comparison data `X`
in terms of the general function `self.g`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>numpy.array or pandas.Series or pandas.DataFrame</i></b>
<p class="attr">
    (# samples, # features) matrix of comparison data.
</p>
<b>bootstrap_samples : <i>scalar</i></b>
<p class="attr">
    Number of bootstrapped samples for computing <code>g</code> of the background data.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>g_comparison : <i>float</i></b>
<p class="attr">
    <em>g(model(X))</em>, where <em>X</em> is the comparison data.
</p>
<b>g_background : <i>float</i></b>
<p class="attr">
    <em>g(model(X_b))</em>, where <em>X_b</em> is the shuffled background data.
</p></td>
</tr>
    </tbody>
</table>





<p class="func-header">
    <i></i> <b>gshap_values</b>(<i>self, X, **kwargs</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/__init__.py#L131">[source]</a>
</p>

Compute G-SHAP values for all features.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>numpy.array or pandas.DataFrame or pandas.Series</i></b>
<p class="attr">
    A (# samples, # features) matrix.
</p>
<b>nsamples : <i>scalar or 'auto', default='auto'</i></b>
<p class="attr">
    Number of samples to draw when approximating G-SHAP values.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>gshap_values : <i>np.array</i></b>
<p class="attr">
    (# features,) vector of G-SHAP values ordered by feature index.
</p></td>
</tr>
    </tbody>
</table>





<p class="func-header">
    <i></i> <b>gshap_value</b>(<i>self, j, X, **kwargs</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/__init__.py#L152">[source]</a>
</p>

Compute the G-SHAP value for feature `j`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>j : <i>scalar or column name</i></b>
<p class="attr">
    The index or column name of the feature of interest.
</p>
<b>X : <i>numpy.array or pandas.DataFrame or pandas.Series</i></b>
<p class="attr">
    A (# samples, # features) matrix.
</p>
<b>nsamples : <i>scalar or 'auto', default='auto'</i></b>
<p class="attr">
    Number of samples to draw when approximating G-SHAP values.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>gshap_value : <i>float</i></b>
<p class="attr">
    Approximated G-SHAP value for feature <code>j</code> (float).
</p></td>
</tr>
    </tbody>
</table>

