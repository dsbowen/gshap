# Technical details

Shapley Additive Explanations (SHAP) measure how important input features are in determining a model's output. The importance of feature \(j\) for model \(f\), \(\phi_j(f)\), is a weighted sum of the feature's contribution to the model's output \(f(x)\) over all possible feature combinations:

$$
    \phi_j(f) = \sum_{S\subseteq \{x_1,...,x_p\}\setminus\{x_j\}}
        \frac{|S|!(p-|S|-1)!}{p!}\big(f(S\cup \{x_j\})-f(S)\big)
$$

Where \(S\) is a subset of features and \(p\) is the number of features in the model. 

In practice, \(f(S)\) is estimated by randomly substituting in values for the remaining features, \(\{x_1,…,x_p\}\setminus S\), from a shuffled background dataset \(X_b\). Suppose we compute the model output for an observation \(f(x)\) and the background dataset \(f(X_b)\). Each SHAP value \(\phi_j\) is the amount of this difference \(f(x)-f(X_b)\) due to feature \(j\).

We can generalize SHAP to compute feature importance for any function \(g\) of the model's output. Define a G-SHAP \(\phi_j^g(f)\) value as:

$$
    \phi_j^g(f) = \sum_{S\subseteq \{x_1,...,x_p\}\setminus\{x_j\}}
        \frac{|S|!(p-|S|-1)!}{p!}\big(g(f(S\cup \{x_j\}))-g(f(S))\big)
$$

G-SHAP values have similar interpretation. Suppose we compute a general function of a model’s output for a sample \(g(f(x))\) and for the shuffled background dataset \(g(f(X_b))\). Each G-SHAP value \(\phi_j^g\) is the amount of this difference \(g(f(x))-g(f(X_b))\) due to feature \(j\).