# Technical

Shapley Additive Explanations (SHAP), derived from Shapley values in game theory, is a popular and mathematically well-grounded example of a feature importance measure. According to SHAP, the importance of feature \(j\) for the output of model \(f\), \(\phi^j(f)\), is a weighted sum of the feature's contribution to the model's output \(f(x_i)\) over all possible feature combinations:

$$
    \phi^j(f) = \sum_{S\subseteq \{x^1,...,x^p\}\setminus\{x^j\}}
        \frac{|S|!(p-|S|-1)!}{p!}\big(f(S\cup \{x^j\})-f(S)\big)
$$

Where \(x^j\) is feature \(j\), \(S\) is a subset of features, and \(p\) is the number of features in the model. 

In practice, \(f(S)\) is estimated by substituting in values for the remaining features, \(\{x^1,…,x^p\}\setminus S\), from a randomly selected observation in a background dataset \(z\in Z\). Suppose we compute the model output for an observation \(f(x_i)\) and a background observation \(f(z)\). Each SHAP value \(\phi^j\) is the amount of this difference \(f(x_i)-f(z)\) due to feature \(j\).

G-SHAP allows us to compute the feature importance of any function \(g\) of a model's output. Define a G-SHAP value \(\phi_g^j(f)\) as:

$$
    \phi_g^j(f) = \sum_{S\subseteq \{x^1,...,x^p\}\setminus\{x^j\}}
        \frac{|S|!(p-|S|-1)!}{p!}\big(g(f,S\cup \{x^j\},\Omega)-g(f,S,\Omega)\big)
$$

Where \(\Omega\) is a set of additional arguments.

G-SHAP values have a similar interpretation as SHAP values. Suppose we compute a function of a model’s output for a sample \(g(f,X,\Omega)\) and a background dataset \(g(f,Z,\Omega)\). Each G-SHAP value \(\phi_g^j\) is the amount of this difference \(g(f,X,\Omega)-g(f,Z,\Omega)\) due to feature \(j\).