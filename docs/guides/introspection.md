# Introspection and Persistence

Fuzzy systems provide a major advantage over black-box deep learning architectures: their internal decision-making structures are fully interpretable. **highFIS** provides dedicated utilities for model introspection (extracting membership parameters, rule tables, and feature importance) and a versioned, secure persistence mechanism.

---

## 1. Model Introspection

Fitted estimators can be introspected to analyze and explain their decision rules. The base estimator interface exposes:

- `inspect()` — a high-level summary dictionary of the fitted model;
- `get_mf_params()` — the membership-function parameters per input feature;
- `feature_importance()` — a normalized importance vector derived from the consequent weights;
- `rule_activation(X)` — the normalized rule firing strengths for given inputs.

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

---

## 2. Model Persistence

highFIS features a native, versioned checkpointing mechanism built on top of PyTorch's serialization engine. Rather than relying on Python `pickle` (which is vulnerable to security exploits and sensitive to package directory shifts), highFIS serialization isolates structural parameters and weights.

> **Warning:** Standard python `pickle` is not recommended for production environments. highFIS checkpointing uses `weights_only=True` PyTorch loading to prevent arbitrary code execution vulnerabilities.

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
