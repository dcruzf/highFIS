# Estimators API

## Module

`highfis.estimators`

## InputConfig

`InputConfig` controls Gaussian initialization per feature.

Fields:

- `name`: feature name.
- `n_mfs`: number of Gaussian membership functions.
- `overlap`: overlap factor controlling membership width.
- `margin`: range padding before center placement.

## HTSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `HTSKClassifier`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `ClassifierMixin`.
- Implements `fit`, `predict`, `predict_proba`, and `score`.
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

- `n_mfs`, `input_configs`
- `epochs`, `learning_rate`
- `rule_base`
- `batch_size`, `shuffle`
- `ur_weight`, `ur_target`
- `consequent_batch_norm`
- `random_state`

### Example

```python
from highfis import HTSKClassifierEstimator

clf = HTSKClassifierEstimator(
    n_mfs=3,
    rule_base="en",
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```
