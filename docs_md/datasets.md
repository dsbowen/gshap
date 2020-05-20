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
</style># Example datasets

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##gshap.datasets.**load_recidivism**

<p class="func-header">
    <i>def</i> gshap.datasets.<b>load_recidivism</b>(<i>return_X_y=False</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/datasets/__init__.py#L32">[source]</a>
</p>

Load the COMPAS recidivism dataset. The purpose of this dataset is to
predict whether a criminal will recidivate within two years of release.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>return_X_y : <i>bool, default=False</i></b>
<p class="attr">
    Indicates whether to return just the X and y matrices, as opposed to the data <code>Bunch</code>.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>bunch : <i>Bunch</i></b>
<p class="attr">
    Object containing the dataframe, X feature matrix, and y target vector. Or, if <code>return_X_y</code>, return (X,y).
</p></td>
</tr>
    </tbody>
</table>



##gshap.datasets.**load_gdp**

<p class="func-header">
    <i>def</i> gshap.datasets.<b>load_gdp</b>(<i>return_X_y=False</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/datasets/__init__.py#L55">[source]</a>
</p>

Load the GDP growth dataset (from FRED data). The purpose of this
dataset is to forecast GDP growth based on macroeconomic variables.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>return_X_y : <i>bool, default=False</i></b>
<p class="attr">
    Indicates whether to return just the X and y matrices, as opposed to the data <code>Bunch</code>.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>bunch : <i>Bunch</i></b>
<p class="attr">
    Object containing the dataframe, X feature matrix, and y target vector. Or, if <code>return_X_y</code>, return (X,y).
</p></td>
</tr>
    </tbody>
</table>



##gshap.datasets.**Bunch**

<p class="func-header">
    <i>class</i> gshap.datasets.<b>Bunch</b>(<i>filename, target</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/gshap/blob/master/gshap/datasets/__init__.py#L78">[source]</a>
</p>

Dataset container.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>filename : <i>str</i></b>
<p class="attr">
    Name of the file in <code>gshap/datasets</code>.
</p>
<b>target : <i>str</i></b>
<p class="attr">
    Name of target variable
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>df : <i>pandas.DataFrame</i></b>
<p class="attr">
    Dataframe containing features and the target variable.
</p>
<b>data : <i>pandas.DataFrame</i></b>
<p class="attr">
    Dataframe containing only the features
</p>
<b>target : <i>pandas.Series</i></b>
<p class="attr">
    Series of the target variable.
</p></td>
</tr>
    </tbody>
</table>



