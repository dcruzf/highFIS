# Introspection and Persistence

Fuzzy systems provide a major advantage over black-box deep learning architectures: their internal decision-making structures are fully interpretable. **highFIS** provides dedicated utilities for model introspection (extracting membership parameters, rule tables, and consequent weights) and a versioned, secure persistence mechanism.

---

## 1. Model Introspection

Fitted estimators can be introspected to analyze and explain their decision rules. The base estimator interface exposes three key methods:

### Membership Function Parameters
`get_mf_params()` returns a serializable dictionary mapping each input feature to a list of its membership function configurations (including their types, centers, and widths).

```python
# Extract membership parameters
mf_params = clf.get_mf_params()
for feature, mfs in mf_params.items():
    print(f"Feature: {feature}")
    for i, mf in enumerate(mfs):
        print(f"  MF {i}: {mf['type']} -> {mf['params']}")
```

### The Rule Base Table
`get_rule_table()` returns the antecedent rule structure as a list of dictionaries. Each dictionary maps the input features to their corresponding fuzzy set index (Membership Function index) for that rule.

```python
# Print rule table
rules = clf.get_rule_table()
for rule in rules:
    rule_id = rule["rule_id"]
    antecedents = [f"{feat} is MF_{rule[feat]}" for feat in clf.feature_names_in_]
    print(f"Rule {rule_id}: IF {' AND '.join(antecedents)} THEN [consequent]")
```

### Consequent Weights
`get_consequent_weights()` extracts the linear coefficients of the consequent equations. For a zero-order TSK system, these represent the constant output values associated with each rule. For a first-order TSK system, these represent the linear combination coefficients.

```python
# Retrieve consequent weights tensor
weights = clf.get_consequent_weights()
if weights is not None:
    print("Consequent weights shape:", weights.shape)
```

---

## 2. Model Persistence

highFIS features a native, versioned checkpointing mechanism built on top of PyTorch's serialization engine. Rather than relying on Python `pickle` (which is vulnerable to security exploits and sensitive to package directory shifts), highFIS serialization isolates structural parameters and weights.

> **Warning:** Standard python `pickle` is not recommended for production environments. highFIS checkpointing uses `weights_only=True` PyTorch loading to prevent arbitrary code execution vulnerabilities.

### Saving a Model

Fitted estimators (both classifiers and regressors) expose a `.save_checkpoint(path)` method:

```python
from highfis import HTSKClassifier

# Fit the classifier
clf = HTSKClassifier(n_mfs=3, random_state=42)
clf.fit(X_train, y_train)

# Save checkpoint to a file
clf.save_checkpoint("models/htsk_iris.pt")
```

### Loading a Model

To restore a saved estimator, call the `.load_checkpoint(path)` classmethod on the corresponding estimator class:

```python
from highfis import HTSKClassifier

# Load and restore the estimator state
loaded_clf = HTSKClassifier.load_checkpoint("models/htsk_iris.pt")

# Predict using the restored estimator
predictions = loaded_clf.predict(X_test)
```

### Checkpoint Validation and Versioning
Behind the scenes, highFIS validates every checkpoint payload. The loader verifies:
1.  **Format Identifier**: Verifies that the file is a valid highFIS payload.
2.  **Format Version**: Ensures backward compatibility by validating the schema version.
3.  **Class Matching**: Prevents restoring a checkpoint created by a different class (e.g., trying to load a regressor checkpoint into a classifier class).
4.  **Schema Completeness**: Validates that all critical components (`estimator_params`, `model_init`, `model_state_dict`, and `fitted_attrs`) are present before reconstructing the estimator.
