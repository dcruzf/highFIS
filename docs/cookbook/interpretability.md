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

Use `inspect()` for a quick overview, `get_mf_params()` / `get_rule_table()` for the
exact antecedents, and `rule_activation()` to see which rules fire for given inputs.
