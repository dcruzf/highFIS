# Inspecting a trained model

A fitted estimator exposes its fuzzy structure so you can interpret it: the rule
base, the membership-function parameters, per-sample rule activations, and a
feature-importance vector derived from the consequent weights.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier

X, y = load_iris(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

# k-means initialization builds a compact CoCo rule base (one rule per cluster),
# which is easier to interpret than the full Cartesian product from "grid".
clf = HTSKClassifier(n_mfs=3, mf_init="kmeans", epochs=20, random_state=0)
clf.fit(X, y)

# High-level summary of the fitted model.
summary = clf.inspect()
print("n_rules:", summary["n_rules"])
print("features:", summary["feature_names"])
print("rule base:", summary["rule_base"])

# Normalized feature importance (sums to 1), from the consequent weights.
importance = clf.feature_importance()
print("importance shape:", None if importance is None else importance.shape)

# Normalized rule activations for the first 5 samples -> shape (5, n_rules).
activations = clf.rule_activation(X[:5])
print("activations shape:", activations.shape)

# Raw membership-function parameters per input feature.
mf_params = clf.get_mf_params()
print("mf params for first feature:", list(mf_params)[0])
```

```text
n_rules: 3
features: ['x1', 'x2', 'x3', 'x4']
rule base: coco
importance shape: (4,)
activations shape: (5, 3)
mf params for first feature: x1
```

## Read the rules

`rules_as_text()` turns the rule base into sentences. Each fuzzy set is named from the
position of its centre, and each rule shows its consequent per class:

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier

X, y = load_iris(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

clf = HTSKClassifier(n_mfs=3, mf_init="kmeans", epochs=20, random_state=0)
clf.fit(X, y)

print(clf.rules_as_text(top_features=2, max_rules=1))
```

```text
Rule 0: IF x1 is low AND x2 is high AND ... (2 more)
        THEN 0: 0.56 - 0.47 * x4 - 0.38 * x3 + ...
             1: -0.55 - 0.70 * x4 - 0.34 * x2 + ...
             2: -0.54 - 0.91 * x2 + 0.47 * x4 + ...
... (2 more rules)
```

`top_features` keeps the conditions on the most important features and the largest
coefficients; `None` shows them all. Fitting on a pandas `DataFrame` with string labels
gives the real feature and class names instead of `x1` and `0`. The exact numbers depend
on the platform.

Use `inspect()` for a quick overview (its `"rule_table"` and `"mf_params"` entries give
the exact antecedents), `get_mf_params()` for the raw membership parameters, and
`rule_activation()` to see which rules fire for given inputs.
