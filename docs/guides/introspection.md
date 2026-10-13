# Introspection and Persistence

Fuzzy systems provide a major advantage over black-box deep learning architectures: their internal decision-making structures are fully interpretable. **highFIS** provides dedicated utilities for model introspection (extracting membership parameters, rule tables, and feature importance) and a versioned, secure persistence mechanism.

---

## 1. Model Introspection

Fitted estimators can be introspected to analyze and explain their decision rules. The base estimator interface exposes:

- `summary()` — a summary of the structure and of the training;
- `inspect()` — a high-level summary dictionary of the fitted model;
- `get_mf_params()` — the membership-function parameters per input feature;
- `feature_importance()` — a normalized importance vector derived from the consequent weights;
- `rule_activation(X)` — the normalized rule firing strengths for given inputs.
- `firing_diagnostics(X)` — a numerical summary of how the rules fire on given inputs.
- `explain(x)` — how the rules produce the prediction of one sample;
- `coverage(X)` — how well the rules cover each sample.

### A Printed Summary

`summary()` returns a text that describes the fitted model: its structure, the training
configuration that `fit` resolved, and how the training ended.

```python
clf.summary()
```

For an `HTSKClassifier` fitted on the Wine data:

```text
HTSKClassifier
  task               classification, 3 classes
  features           13 seen in fit, 13 used by the model
  rules              3 (rule base: coco)
  fuzzy sets         3 per feature, GaussianMF
  firing strength    RuleLayer, T-norm GMeanTNorm
  defuzzifier        SoftmaxLogDefuzzifier
  consequents        ClassificationConsequentLayer
  parameters         204 trainable (78 antecedent, 126 consequent)
Training (batch size 60)
  phase     epochs  rate  loss first -> last  accuracy
  training     100  0.01    1.0056 -> 0.0045    1.0000
```

The learning rate and the batch size are the values used, also when the estimator was
built with `"auto"`. The families that train in phases (DG-TSK, DG-ALETSK, FSRE-ADATSK)
have one line per phase, and "features used by the model" counts the features they kept.

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
clf.rules_as_text(top_features=2, max_rules=1)
```

- The conditions of a rule are written in full; when more than half of the features use
  the same fuzzy set, the rule is written as the features that differ followed by
  `the other N features are low`.
- `top_features` limits each consequent to the coefficients with the largest magnitude;
  `None` shows them all.
- `width` is the width of the text; longer lines are continued on the next one.
- `labels=["cold", "mild", "hot"]` replaces the default names for the features that have
  that many sets.
- `max_rules` limits the number of rules written.
- Feature names are used when the model was fitted on a pandas `DataFrame`; class names
  come from `classes_`.

The coefficients apply to the features as they were passed to `fit`, so they are on the
scaled features when the data were scaled first.

### Explaining One Prediction

`explain(x)` describes how the rules produce the prediction of one sample. The output of
a TSK system is the sum, over the rules, of the normalized weight of the rule times the
value of its consequent, so each rule has a contribution that can be read on its own.

```python
clf.explain(X_test[0], top_rules=2)
```

For the model above and one test sample:

```text
Prediction: class_2
Output, the sum of the contributions of the 3 rules: class_0: -3.26,
    class_1: -3.12, class_2: +3.38
Rule 1, weight 0.99: contributes class_0: -3.24, class_1: -3.13, class_2: +3.41
    IF x1 is medium AND x2 is high AND x3 is high AND x4 is medium
       AND x5 is medium AND x6 is low AND x7 is low AND x8 is high AND x9 is low
       AND x10 is high AND x11 is low AND x12 is low AND x13 is medium
Rule 0, weight 0.01: contributes class_0: -0.02, class_1: +0.02, class_2: -0.02
    IF x1 is low AND x2 is low AND x3 is low AND x4 is high AND x5 is low
       AND x6 is medium AND x7 is high AND x8 is medium AND x9 is high
       AND x10 is low AND x11 is high AND x12 is high AND x13 is low
1 other rule, weight 0.00 in all: contributes class_0: +0.01, class_1: -0.01,
    class_2: -0.01
```

- For a classifier the output has one score per class, and the predicted class is the
  one with the largest score. For a regressor it is the predicted value.
- `top_rules` sets how many rules are described, from the largest weight; the others are
  summed in one line. `None` describes every rule.
- `labels` works as in `rules_as_text()`, and the conditions of each rule are written in
  full. A rule whose weight rounds to zero is counted with the other rules.

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
than half of the samples. One rule taking almost all the weight raises no warning: some
families concentrate the weight on high-dimensional data, and `dominated_fraction` and
`mean_firing` show it when you ask. Models with a single rule are not checked. To silence
the warning:

```python
import warnings

from highfis import DegenerateFiringWarning

warnings.simplefilter("ignore", DegenerateFiringWarning)
```

### Coverage

`coverage(X)` says, for each sample, how well the best-matching rule covers it. The
match of a sample with a rule is the mean of its membership degrees in the fuzzy sets of
the rule: 1 at the centre of every set, close to 0 far from all of them. It does not
depend on the number of features nor on the T-norm of the family.

```python
import numpy as np

inside = clf.coverage(X_train)
print(f"training data: min {inside.min():.2f}, median {np.median(inside):.2f}")
print(f"shifted data:  median {np.median(clf.coverage(X_test + 2.0)):.2f}")
```

What to look for: compare new samples with the training data. A sample whose coverage
is well below the lowest values seen in training lies outside the region the rules were
built on, and the model extrapolates there. The families whose membership function has
a positive lower bound (ADPTSK, ADMTSK, DombiTSK, AYATSK) cannot go below that bound.

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
