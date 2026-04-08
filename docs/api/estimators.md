# Estimators API

## Module

`highfis.estimators`

## InputConfig

`InputConfig` controls per-feature Gaussian setup used by `mf_init="grid"`.

Fields:

- `name`: feature name.
- `n_mfs`: number of Gaussian membership functions.
- `overlap`: overlap factor controlling membership width.
- `margin`: range padding before center placement.

When `mf_init="kmeans"` (default), `InputConfig` is used only for feature names
(if provided), while centers and sigmas are estimated from k-means clusters.

## HTSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `HTSKClassifier`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `ClassifierMixin`.
- Implements `fit`, `predict`, `predict_proba`, and `score`.
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

- `n_mfs`, `input_configs`
- `mf_init` (`"kmeans"` default, `"grid"` optional)
- `sigma_scale` (k-means sigma scaling factor, paper's $h$)
- `epochs`, `learning_rate`
- `rule_base`
- `batch_size`, `shuffle`
- `ur_weight`, `ur_target`
- `consequent_batch_norm`
- `random_state`
- `patience` (early-stopping patience in epochs)
- `validation_data` (tuple `(x_val, y_val)` for early stopping by accuracy)
- `weight_decay` (consequent parameter weight decay for AdamW, default $10^{-8}$)

### Initialization Modes

- `mf_init="kmeans"` (default):
    runs k-means with `n_clusters=n_mfs`; for each rule $r$ and feature $d$,
    the Gaussian center is the centroid coordinate $m_{r,d}$ and sigma is the
    within-cluster spread scaled by `sigma_scale`.
- `mf_init="grid"`:
    keeps the original per-feature grid initialization controlled by
    `InputConfig(n_mfs, overlap, margin)`.

Default `rule_base` depends on the initialization mode:

- `"coco"` when `mf_init="kmeans"` (one rule per cluster)
- `"cartesian"` when `mf_init="grid"`

### Example

```python
from highfis import HTSKClassifierEstimator

clf = HTSKClassifierEstimator(
    n_mfs=3,
    mf_init="kmeans",      # default
    sigma_scale=1.0,        # paper-recommended default for HTSK
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```

Grid-based initialization remains available:

```python
clf = HTSKClassifierEstimator(
    n_mfs=3,
    mf_init="grid",
    rule_base="cartesian",
)
```
