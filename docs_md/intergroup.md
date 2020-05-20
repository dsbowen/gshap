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
</style>#Intergroup differences

For examples and interpretation, see my notebook on [intergroup difference explanations](https://github.com/dsbowen/gshap/blob/master/intergroup_difference.ipynb).

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##gshap.intergroup.**IntergroupDifference**

<p class="func-header">
    <i>class</i> gshap.intergroup.<b>IntergroupDifference</b>(<i>group, distance='absolute_mean_distance'</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/intergroup.py#L13">[source]</a>
</p>

This class measures the distance between distributions of predicted
outcomes for different groups.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Paramters:</b></td>
    <td class="field-body" width="100%"><b>group : <i>numpy.array or pandas.Series</i></b>
<p class="attr">
    (# observations,) array of boolean or binary values indicating group membership.
</p>
<b>distance : <i>callable or str, default='absolute_mean_distance'</i></b>
<p class="attr">
    Takes two vectors of model output for the outgroup and ingroup. Output vectors will usually be (# outgroup,) and (# ingroup,), or (# outgroup, # classes) and (# ingroup, # classes). <code>distance</code> returns a scalar measure of intergroup difference, such as the absolute difference between group means. If input as a string, <code>distance</code> is used as a key to look up built-in distance functions.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>group : <i>numpy.array</i></b>
<p class="attr">
    Set from the <code>group</code> parameter. If the parameter is passed as a <code>pandas.Series</code>, it is automatically converted in a <code>numpy.array</code>.
</p>
<b>distance : <i>callable or str</i></b>
<p class="attr">
    Set from the <code>distance</code> parameter.
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
import gshap
from gshap.datasets import load_recidivism
from gshap.intergroup import IntergroupDifference

from sklearn.svm import SVC

recidivism = load_recidivism()
X, y = recidivism.data, recidivism.target
clf = SVC().fit(X,y)

g = IntergroupDifference(group=X['black'], distance='relative_mean_distance')
explainer = gshap.KernelExplainer(clf.predict, X, g)
explainer.gshap_values(X, nsamples=10)
```

Out:

```
array([ 0.01335252,  0.24884556,  0.00132373, -0.0025238 , -0.00151837,
    0.40453822,  0.01636782,  0.07666043, -0.00056414,  0.00966583])
```

####Methods



<p class="func-header">
    <i></i> <b>__call__</b>(<i>self, output</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/intergroup.py#L73">[source]</a>
</p>

Compute distance measure between groups.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>ouput : <i>numpy.array or pandas.Series</i></b>
<p class="attr">
    Model output, usually a (# observations,) or (# observations, # classes) vector.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>distance : <i>scalar</i></b>
<p class="attr">
    Measure of the distance between the distributions of predicted outputs for outgroup and ingroup observations.
</p></td>
</tr>
    </tbody>
</table>



##gshap.intergroup.**absolute_mean_distance**

<p class="func-header">
    <i>def</i> gshap.intergroup.<b>absolute_mean_distance</b>(<i>out_0, out_1</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/intergroup.py#L94">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>out_0 : <i>np.array</i></b>
<p class="attr">
    (# observations,) vector of model outputs for outgroup observations.
</p>
<b>out_1 : <i>np.array</i></b>
<p class="attr">
    (# observations,) vector of model outputs for ingroup observations.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>distance : <i>scalar</i></b>
<p class="attr">
    out_1.mean() - out_0.mean()
</p></td>
</tr>
    </tbody>
</table>



##gshap.intergroup.**relative_mean_distance**

<p class="func-header">
    <i>def</i> gshap.intergroup.<b>relative_mean_distance</b>(<i>out_0, out_1</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/intergroup.py#L112">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>out_0 : <i>np.array</i></b>
<p class="attr">
    (# observations,) vector of model outputs for outgroup observations.
</p>
<b>out_1 : <i>np.array</i></b>
<p class="attr">
    (# observations,) vector of model outputs for ingroup observations.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>distance : <i>scalar</i></b>
<p class="attr">
    out_1.mean() / out_0.mean() - 1
</p></td>
</tr>
    </tbody>
</table>

