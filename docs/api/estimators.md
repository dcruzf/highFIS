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

## HTSKRegressorEstimator

A scikit-learn compatible regressor wrapper around `HTSKRegressor`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `RegressorMixin`.
- Implements `fit`, `predict`, and `score` ($R^2$).
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

Same hyperparameters as `HTSKClassifierEstimator` (see above), except:

- No `n_classes` parameter — output is scalar.
- `score()` returns $R^2$ instead of accuracy.

### Example

```python
from highfis import HTSKRegressorEstimator

reg = HTSKRegressorEstimator(
    n_mfs=3,
    mf_init="kmeans",
    sigma_scale=1.0,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
reg.fit(X_train, y_train)
r2 = reg.score(X_test, y_test)
```

## TSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `TSKClassifier`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `ClassifierMixin`.
- Implements `fit`, `predict`, `predict_proba`, and `score`.
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

Same hyperparameters as `HTSKClassifierEstimator` (see above).

The model uses `SumBasedDefuzzifier` (w / Σw) instead of `SoftmaxLogDefuzzifier`.

### Example

```python
from highfis import TSKClassifierEstimator

clf = TSKClassifierEstimator(
    n_mfs=3,
    mf_init="kmeans",
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```

## TSKRegressorEstimator

A scikit-learn compatible regressor wrapper around `TSKRegressor`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `RegressorMixin`.
- Implements `fit`, `predict`, and `score` ($R^2$).
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

Same hyperparameters as `HTSKRegressorEstimator` (see above), with `SumBasedDefuzzifier`.

### Example

```python
from highfis import TSKRegressorEstimator

reg = TSKRegressorEstimator(
    n_mfs=3,
    mf_init="kmeans",
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
reg.fit(X_train, y_train)
r2 = reg.score(X_test, y_test)
```

## DombiTSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `DombiTSKClassifier`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `ClassifierMixin`.
- Implements `fit`, `predict`, `predict_proba`, and `score`.
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

Same hyperparameters as `HTSKClassifierEstimator` (see above).

### Example

```python
from highfis import DombiTSKClassifierEstimator

clf = DombiTSKClassifierEstimator(
    n_mfs=3,
    mf_init="kmeans",
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```

## DombiTSKRegressorEstimator

A scikit-learn compatible regressor wrapper around `DombiTSKRegressor`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `RegressorMixin`.
- Implements `fit`, `predict`, and `score` ($R^2$).
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

Same hyperparameters as `HTSKRegressorEstimator` (see above), with `SumBasedDefuzzifier`.

### Example

```python
from highfis import DombiTSKRegressorEstimator

reg = DombiTSKRegressorEstimator(
    n_mfs=3,
    mf_init="kmeans",
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
reg.fit(X_train, y_train)
r2 = reg.score(X_test, y_test)
```

## AdaTSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `AdaTSKClassifier`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `ClassifierMixin`.
- Implements `fit`, `predict`, `predict_proba`, and `score`.
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

Same hyperparameters as `HTSKClassifierEstimator` (see above), plus:

- `lambda_init`: positive initial value for adaptive Dombi shape parameters.

### Example

```python
from highfis import AdaTSKClassifierEstimator

clf = AdaTSKClassifierEstimator(
    n_mfs=3,
    mf_init="kmeans",
    lambda_init=1.0,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```

## AdaTSKRegressorEstimator

A scikit-learn compatible regressor wrapper around `AdaTSKRegressor`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `RegressorMixin`.
- Implements `fit`, `predict`, and `score` ($R^2$).
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

Same hyperparameters as `HTSKRegressorEstimator` (see above), plus:

- `lambda_init`: positive initial value for adaptive Dombi shape parameters.

### Example

```python
from highfis import AdaTSKRegressorEstimator

reg = AdaTSKRegressorEstimator(
    n_mfs=3,
    mf_init="kmeans",
    lambda_init=1.0,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
reg.fit(X_train, y_train)
r2 = reg.score(X_test, y_test)
```

## LogTSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `LogTSKClassifier`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `ClassifierMixin`.
- Implements `fit`, `predict`, `predict_proba`, and `score`.
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

Same hyperparameters as `HTSKClassifierEstimator` (see above).

The model uses `LogSumDefuzzifier` (softmax(log(w)/τ)) for log-space normalization.

### Example

```python
from highfis import LogTSKClassifierEstimator

clf = LogTSKClassifierEstimator(
    n_mfs=3,
    mf_init="kmeans",
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```

## LogTSKRegressorEstimator

A scikit-learn compatible regressor wrapper around `LogTSKRegressor`.

### sklearn Compatibility

- Inherits `BaseEstimator` and `RegressorMixin`.
- Implements `fit`, `predict`, and `score` ($R^2$).
- Works with `Pipeline`, `GridSearchCV`, and cross-validation tools.

### Core Hyperparameters

Same hyperparameters as `HTSKRegressorEstimator` (see above), with `LogSumDefuzzifier`.

### Example

```python
from highfis import LogTSKRegressorEstimator

reg = LogTSKRegressorEstimator(
    n_mfs=3,
    mf_init="kmeans",
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
reg.fit(X_train, y_train)
r2 = reg.score(X_test, y_test)
```
