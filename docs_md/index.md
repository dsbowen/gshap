# Generalized Shapley Additive Explanations

Generalized Shapley Additive Explanations (G-SHAP) is a technique in explainable AI for answering broad questions in machine learning.

## Applications

This is just a small sample of the questions G-SHAP can answer.

### General classification and regression

Suppose we have a black-box model which diagnoses patients with COVID-19, the flu, or a common cold based on their symptoms. Existing explanatory methods can tell us why our model diagnosed a patient with COVID-19. G-SHAP can answer broader questions, such as *how do the symptoms which distinguish COVID-19 from the flu differ from those which distinguish COVID-19 from the common cold?*.

Full analysis [here](https://github.com/dsbowen/gshap/blob/master/classification.ipynb).

### Intergroup differences

Suppose we have a black-box model which predicts a criminal’s risk of recidivism to determine whether they are eligible for parole. Existing explanatory methods can tell us why our model predicted that a criminal has a high recidivism risk. G-SHAP can answer broader questions, such as *why does our model predict that Black criminals have higher recidivism rates than White criminals?*.

Full analysis [here](https://github.com/dsbowen/gshap/blob/master/intergroup_difference.ipynb).

### Model performance and failure

Suppose we have a black-box model which forecasts GDP growth based on macroeconomic variables. Existing explanatory methods can tell us why our model forecast 3% GDP growth in a given year. G-SHAP can answer broader questions, such as *why did our model fail to forecast the 2008-2009 financial crisis?*.

Full analysis [here](https://github.com/dsbowen/gshap/blob/master/model_failure_regression.ipynb).

## Installation

```
$ pip install gshap
```

## Quickstart

Here we train a support vector classifier to predict whether a criminal will recidivate within two years of release from prison. We use G-SHAP to ask why our model predicts that Black criminals are more likely to recidivate than non-Black criminals.

```python
import gshap
from gshap.datasets import load_recidivism
from gshap.intergroup import IntergroupDifference

from sklearn.svm import SVC

recidivism = load_recidivism()
X, y = recidivism.data, recidivism.target
clf = SVC().fit(X, y)

g = IntergroupDifference(group=X['black'], distance='relative_mean_distance')
explainer = gshap.KernelExplainer(clf.predict, X, g)
explainer.gshap_values(X, nsamples=10)
```

Out:

```
array([ 0.01335252,  0.24884556,  0.00132373, -0.0025238 , -0.00151837,
    0.40453822,  0.01636782,  0.07666043, -0.00056414,  0.00966583])
```

The sum of the G-SHAP values is the relative difference in predicted recidivism rates. The model predicts that Black criminals are 75% more likely to recidivate. 

The variables most responsible for this difference are number of prior convictions (index 5; 40%), age (index 1; 25%), and race (index 7; 8%).

## Citation

```
@software{bowen2020gshap,
  author = {Dillon Bowen},
  title = {Generalized Shapley Additive Explanations},
  url = {https://dsbowen.github.io/gshap/},
  date = {2020-05-19},
}
```

## License

Users must cite G-SHAP in any publications which use this software.

G-SHAP is licensed with the MIT [License](https://github.com/dsbowen/gshap/blob/master/LICENSE).