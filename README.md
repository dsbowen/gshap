# Generalized SHAP

**Generalized Shapley Additive Explanations (G-SHAP)** generalizes [SHAP](https://github.com/slundberg/shap) to answer broader questions in machine learning.

Its uses include:

1. **General classification explanations**. What features distinguish class $c_0$ from class $c_1$?
2. **Intergroup differences**. Why does our model make different predictions for different groups of observations?
3. **Model failure**. Why does our model perform poorly on sample $x$?

## Example

Suppose we have a `model` which takes a $n\times p$ feature matrix and returns a $n\times q$ output matrix. We also have a function `g` which takes the output matrix and returns a scalar. We can use G-SHAP to explain why the output of function `g` differs for a sample `x` relative to a background dataset `X_b` as follows:

```python
explainer = gshap.KernelExplainer(model, X_b, g)
gshap_values = explainer.gshap_values(x)
```

`gshap_values` is a $p\times 1$ vector of feature importances. Each G-SHAP value `gshap_values[j]` is the amount of the difference `g(model(x))-g(model(X_b))` explained by feature $j$.

## Documentation

You can find the latest documentation at https://dsbowen.github.io/gshap.

## License

Publications which use this software should include the following citation:

Bowen, D.S. (2020). Generalized SHAP: Methods for answering broader questions in machine learning. https://dsbowen.github.io/gshap.

BibTex:

```
@software{bowen2020gshap,
  author = {Dillon Bowen},
  title = {Generalized SHAP: Methods for answering broader questions in machine learning},
  url = {https://dsbowen.github.io/gshap},
  version = {0.0.1},
  date = {2020-04-28},
}
```

This project is licensed under the MIT License [LICENSE](https://github.com/dsbowen/gshap/blob/master/LICENSE).