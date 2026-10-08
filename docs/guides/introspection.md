# Introspection and Persistence

Fuzzy systems provide a major advantage over black-box deep learning architectures: their internal decision-making structures are fully interpretable. **highFIS** provides dedicated utilities for model introspection (extracting membership parameters, rule tables, and feature importance) and a versioned, secure persistence mechanism.

---

## 1. Model Introspection

Fitted estimators can be introspected to analyze and explain their decision rules. The base estimator interface exposes:

- `inspect()` — a high-level summary dictionary of the fitted model;
- `get_mf_params()` — the membership-function parameters per input feature;
- `feature_importance()` — a normalized importance vector derived from the consequent weights;
- `rule_activation(X)` — the normalized rule firing strengths for given inputs.
- `firing_diagnostics(X)` — a numerical summary of how the rules fire on given inputs.

### High-level summary

`inspect()` returns a dictionary describing the fitted structure. Its keys are
`n_rules`, `n_inputs`, `feature_names`, `rule_base`, `defuzzifier_type`, `mf_params`,
and `rule_table`.

```python
summary = clf.inspect()
print("Rules:", summary["n_rules"])
print("Features:", summary["feature_names"])
print("Rule base:", summary["rule_base"])
```

### Membership Function Parameters

`get_mf_params()` returns a serializable dictionary mapping each input feature to a
list of its membership-function configurations. Each entry has a `type` key naming the
MF class, plus that type's own parameter keys (for example a `GaussianMF` adds `mean`
and `sigma`). The same dictionary is available as `summary["mf_params"]`.

```python
mf_params = clf.get_mf_params()
for feature, mfs in mf_params.items():
    print(f"Feature: {feature}")
    for i, mf in enumerate(mfs):
        params = {k: v for k, v in mf.items() if k != "type"}
        print(f"  MF {i}: {mf['type']} -> {params}")
```

### The Rule Base Table

The antecedent rule structure is available as `summary["rule_table"]`: a list of
dictionaries, one per rule. Each dictionary carries a `rule_id` and maps every input
feature to the index of the membership function it uses in that rule.

```python
summary = clf.inspect()
for rule in summary["rule_table"]:
    rule_id = rule["rule_id"]
    antecedents = [f"{feat} is MF_{rule[feat]}" for feat in summary["feature_names"]]
    print(f"Rule {rule_id}: IF {' AND '.join(antecedents)} THEN [consequent]")
```

The same table is returned directly by `clf.get_rule_table()`.

### Rules as Text

`rules_as_text()` writes the rule base as `IF ... THEN ...` sentences. The fuzzy sets of
each feature are named from the order of their centres (`low`, `medium`, `high` for three
sets), and the consequent of each rule is its linear function of the features, per class
for a classifier.

```python
print(clf.rules_as_text(top_features=2, max_rules=1))
```

- `top_features` limits each rule to the conditions on the most important features and
  to the coefficients with the largest magnitude; `None` shows everything.
- `labels=["cold", "mild", "hot"]` replaces the default names for the features that have
  that many sets.
- `max_rules` limits the number of rules written.
- Feature names are used when the model was fitted on a pandas `DataFrame`; class names
  come from `classes_`.

The coefficients apply to the features as they were passed to `fit`, so they are on the
scaled features when the data were scaled first.

### Feature Importance

`feature_importance()` returns a normalized vector (summing to 1) that ranks the input
features by their contribution to the consequent, or `None` when the model has no
first-order consequent to read it from.

```python
importance = clf.feature_importance()
if importance is not None:
    for feat, score in zip(clf.inspect()["feature_names"], importance):
        print(f"{feat}: {score:.3f}")
```

### Consequent Parameters

A first-order rule computes `score_r^c(x) = b_{r,c} + sum_d w_{r,c,d} x_d`. Both halves are
available on the estimator as NumPy arrays: `get_consequent_weights()` returns `w` and
`get_consequent_bias()` returns the intercept `b`.

```python
weights = clf.get_consequent_weights()  # (rules, classes, features)
bias = clf.get_consequent_bias()        # (rules, classes)
if weights is not None and bias is not None:
    print("rule 0, class 0 intercept:", float(bias[0, 0]))
    print("rule 0, class 0 slopes:", weights[0, 0].tolist())
```

### Rule-Firing Diagnostics

The normalized rule weights say how each prediction is shared among the rules. When they
stop depending on the sample, the rules no longer partition the input space and the model
is a single linear model, whatever its number of rules. `firing_diagnostics(X)` measures
this:

| Entry | Meaning |
|---|---|
| `uniform_fraction` | Fraction of samples on which every rule has the same weight. This is what the underflow of a product of many membership degrees produces. |
| `dominated_fraction` | Fraction of samples on which one rule has at least 99% of the weight (`dominance` argument). |
| `effective_rules`, `effective_rules_min` | Mean and minimum over the samples of the exponential of the entropy of the weights: 1 when one rule decides, `n_rules` when all weigh the same. |
| `mean_firing` | Mean weight of each rule. |
| `never_firing_rules` | Indices of the rules whose weight stays below `never` (default `1e-6`) on every sample. |
| `non_finite_fraction` | Fraction of samples with a weight that is not finite. |
| `n_samples`, `n_rules` | Size of the weights that were summarized. |

A high `dominated_fraction` is not a defect in itself: it means the rules split the input
space sharply. A high `uniform_fraction` is one.

```python
import numpy as np

from highfis import HTSKClassifier, TSKClassifier

rng = np.random.default_rng(0)
X_wide = rng.random((60, 2000)).astype(np.float32)
y_wide = (X_wide[:, 0] > 0.5).astype(int)

product = TSKClassifier(n_mfs=3, epochs=5, random_state=0).fit(X_wide, y_wide)  # warns
print(product.firing_diagnostics(X_wide)["uniform_fraction"])  # 1.0

stable = HTSKClassifier(n_mfs=3, epochs=5, random_state=0).fit(X_wide, y_wide)
print(stable.firing_diagnostics(X_wide)["uniform_fraction"])  # 0.0
```

At the end of `fit`, the same check runs on the training data (at most 2000 rows) and a
`highfis.DegenerateFiringWarning` is raised when every rule has the same weight on more
than half of the samples, or when one rule takes more than 99% of the weight on average.
Models with a single rule are not checked. To silence it:

```python
import warnings

from highfis import DegenerateFiringWarning

warnings.simplefilter("ignore", DegenerateFiringWarning)
```

### Selected Features and Gates

DG-TSK, DG-ALETSK and FSRE-ADATSK switch features and rules off with gates and then
remove them from the model. Three accessors describe the result:

- `selected_features_`: the column indices, in the data seen in `fit`, of the features
  the model uses. For the other families it lists every column.
- `get_feature_gates()`: one gate value in `[0, 1]` per selected feature.
- `get_rule_gates()`: one gate value in `[0, 1]` per rule.

The two gate accessors return `None` for families without gates.

```python
import numpy as np

from highfis import DGTSKClassifier

rng = np.random.default_rng(0)
X_many = rng.random((120, 12))
y_many = (X_many[:, 0] + X_many[:, 3] > 1.0).astype(int)

gated = DGTSKClassifier(n_mfs=3, dg_epochs=10, finetune_epochs=10, random_state=0).fit(X_many, y_many)

print("features kept:", gated.selected_features_.tolist())
print("rules kept:", len(gated.get_rule_gates()))
```

---

## 2. Model Persistence

highFIS features a native, versioned checkpointing mechanism built on top of PyTorch's serialization engine. Rather than relying on Python `pickle` (which is vulnerable to security exploits and sensitive to package directory shifts), highFIS serialization isolates structural parameters and weights. It uses `weights_only=True` PyTorch loading, so loading a checkpoint does not execute code.

A checkpoint stores the estimator alone. To keep the preprocessing together with the model, put both in a scikit-learn `Pipeline` and save the pipeline with `joblib`; see [Save a pipeline and use it on new data](../cookbook/persistence-and-cache.md#save-a-pipeline-and-use-it-on-new-data).

> **Warning:** `joblib` and `pickle` can execute arbitrary code when loading. Use them only for files you created or that come from a source you trust.

### Saving a Model

Fitted estimators (both classifiers and regressors) expose a `.save(path)` method:

```{.python notest}
from highfis import HTSKClassifier

# Fit the classifier
clf = HTSKClassifier(n_mfs=3, random_state=42)
clf.fit(X_train, y_train)

# Save checkpoint to a file
clf.save("models/htsk_iris.pt")
```

### Loading a Model

To restore a saved estimator, call the `.load(path)` classmethod on the corresponding estimator class:

```{.python notest}
from highfis import HTSKClassifier

# Load and restore the estimator state
loaded_clf = HTSKClassifier.load("models/htsk_iris.pt")

# Predict using the restored estimator
predictions = loaded_clf.predict(X_test)
```

### Checkpoint Validation and Versioning
Behind the scenes, highFIS validates every checkpoint payload. The loader verifies:
1.  **Format Identifier**: Verifies that the file is a valid highFIS payload.
2.  **Format Version**: Ensures backward compatibility by validating the schema version.
3.  **Class Matching**: Prevents restoring a checkpoint created by a different class (e.g., trying to load a regressor checkpoint into a classifier class).
4.  **Schema Completeness**: Validates that all critical components (`estimator_params`, `model_init`, `model_state_dict`, and `fitted_attrs`) are present before reconstructing the estimator.
