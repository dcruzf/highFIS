# Estimators API

## Module

`highfis.estimators`

This module provides scikit-learn compatible wrappers for highFIS TSK models.
Each estimator implements the standard `fit`/`predict` interface and builds an
underlying `BaseTSK` model from Gaussian membership functions.

## InputConfig

`InputConfig` configures per-feature Gaussian MF construction for
`mf_init="grid"`.

Fields:

- `name`: feature name.
- `n_mfs`: number of Gaussian membership functions.
- `overlap`: controls the spacing of grid-initialized membership widths.
- `margin`: padding applied to the feature range before center placement.

When `mf_init="kmeans"`, `InputConfig` is only used to name features; centers
and sigmas are estimated from k-means cluster centroids.

## Common hyperparameters

Most estimators share these common settings:

- `input_configs`: optional list of `InputConfig`; must match the number of features.
- `n_mfs`: number of MFs per feature for grid initialization or number of k-means
  clusters for clustering-based initialization.
- `mf_init`: initialization mode, either `"kmeans"` or `"grid"`.
- `sigma_scale`: scale factor for k-means sigma initialization; accepts a float or
  `"auto"`.
- `random_state`: RNG seed for deterministic initialization.
- `epochs`: number of training epochs.
- `learning_rate`: optimizer learning rate.
- `verbose`: whether to print training progress.
- `rule_base`: explicit rule base type; defaults to `"coco"` for k-means and
  `"cartesian"` for grid initialization.
- `batch_size`: training batch size.
- `shuffle`: whether to shuffle training samples each epoch.
- `ur_weight`: uncertainty regularization weight.
- `ur_target`: optional target for uncertainty regularization.
- `consequent_batch_norm`: whether to apply batch normalization to consequent layers.
- `patience`: early stopping patience.
- `validation_data`: optional tuple `(x_val, y_val)` for validation during training.
- `weight_decay`: weight decay for consequent parameters.

## Initialization modes

- `mf_init="kmeans"` (default):
  - runs k-means with `n_clusters=n_mfs`.
  - Gaussian centers are set to cluster centroid coordinates.
  - Sigma is initialized from within-cluster spread scaled by `sigma_scale`.
- `mf_init="grid"`:
  - builds per-feature Gaussian MFs over a regular grid.
  - uses `InputConfig` values to control center placement and overlap.

## Base estimator behavior

### `_BaseClassifierEstimator`

- Builds a classifier-specific `BaseTSK` model in `_build_model`.
- Encodes labels with `LabelEncoder`.
- Supports `fit`, `predict_proba`, `predict`, and `score`.
- `save(path)` persists estimator parameters, model state, and fitted metadata.
- `load(path)` reconstructs the estimator and model from a checkpoint.

### `_BaseRegressorEstimator`

- Builds a regressor-specific `BaseTSK` model in `_build_model`.
- Supports `fit`, `predict`, and `score`.
- `save(path)` persists estimator parameters, model state, and fitted metadata.
- `load(path)` reconstructs the estimator and model from a checkpoint.

## HTSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `HTSKClassifier`.

### Summary

- Uses `HTSKClassifier` as the underlying model.
- Supports grid and k-means MF initialization.
- Uses `cartesian` rule base for grid initialization and `coco` for k-means by
  default.
- Exposes shared estimator hyperparameters plus regularization and validation.

### Example

```python
from highfis import HTSKClassifierEstimator

clf = HTSKClassifierEstimator(
    n_mfs=3,
    mf_init="kmeans",
    sigma_scale=1.0,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```

## HTSKRegressorEstimator

A scikit-learn compatible regressor wrapper around `HTSKRegressor`.

### Summary

- Uses `HTSKRegressor` as the underlying model.
- Shares the same hyperparameters and initialization behavior as the
  classifier wrapper.

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

### Summary

- Uses `TSKClassifier` as the underlying model.
- Implements `fit`, `predict_proba`, `predict`, and `score`.

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

### Summary

- Uses `TSKRegressor` as the underlying model.
- Implements `fit`, `predict`, and `score`.

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

### Summary

- Uses `DombiTSKClassifier` as the underlying model.
- Shares the same estimator interface and hyperparameters as other
  classifier wrappers.

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

### Summary

- Uses `DombiTSKRegressor` as the underlying model.
- Shares the same estimator interface and hyperparameters as other
  regressor wrappers.

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

### Summary

- Adds `lambda_init` to initialize the adaptive Dombi shape parameter.
- Validates that `lambda_init > 0`.
- Uses `AdaTSKClassifier` as the underlying model.

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

### Summary

- Adds `lambda_init` to initialize the adaptive Dombi shape parameter.
- Validates that `lambda_init > 0`.
- Uses `AdaTSKRegressor` as the underlying model.

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

## FSREAdaTSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `FSREAdaTSKClassifier`.

### Summary

- Adds `use_en_frb` to enable the enhanced fuzzy rule base.
- Adds `lambda_init` for adaptive Dombi shape initialization.
- Uses `FSREAdaTSKClassifier` as the underlying model.

### Example

```python
from highfis import FSREAdaTSKClassifierEstimator

clf = FSREAdaTSKClassifierEstimator(
    n_mfs=3,
    mf_init="kmeans",
    lambda_init=1.0,
    use_en_frb=True,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```

## FSREAdaTSKRegressorEstimator

A scikit-learn compatible regressor wrapper around `FSREAdaTSKRegressor`.

### Summary

- Adds `use_en_frb` to enable the enhanced fuzzy rule base.
- Adds `lambda_init` for adaptive Dombi shape initialization.
- Uses `FSREAdaTSKRegressor` as the underlying model.

### Example

```python
from highfis import FSREAdaTSKRegressorEstimator

reg = FSREAdaTSKRegressorEstimator(
    n_mfs=3,
    mf_init="kmeans",
    lambda_init=1.0,
    use_en_frb=True,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
reg.fit(X_train, y_train)
r2 = reg.score(X_test, y_test)
```

## DGALETSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `DGALETSKClassifier`.

### Summary

- Inherits from `FSREAdaTSKClassifierEstimator`.
- Adds `lambda_init` and `use_en_frb`.
- Uses `DGALETSKClassifier` as the underlying model.

### Example

```python
from highfis import DGALETSKClassifierEstimator

clf = DGALETSKClassifierEstimator(
    n_mfs=3,
    mf_init="kmeans",
    lambda_init=1.0,
    use_en_frb=True,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```

## DGALETSKRegressorEstimator

A scikit-learn compatible regressor wrapper around `DGALETSKRegressor`.

### Summary

- Inherits from `FSREAdaTSKRegressorEstimator`.
- Adds `lambda_init` and `use_en_frb`.
- Uses `DGALETSKRegressor` as the underlying model.

### Example

```python
from highfis import DGALETSKRegressorEstimator

reg = DGALETSKRegressorEstimator(
    n_mfs=3,
    mf_init="kmeans",
    lambda_init=1.0,
    use_en_frb=True,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
reg.fit(X_train, y_train)
r2 = reg.score(X_test, y_test)
```

## DGTSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `DGTSKClassifier`.

### Summary

- Adds `use_en_frb` to enable the enhanced fuzzy rule base.
- Uses `DGTSKClassifier` as the underlying model.

### Example

```python
from highfis import DGTSKClassifierEstimator

clf = DGTSKClassifierEstimator(
    n_mfs=3,
    mf_init="kmeans",
    use_en_frb=True,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```

## DGTSKRegressorEstimator

A scikit-learn compatible regressor wrapper around `DGTSKRegressor`.

### Summary

- Adds `use_en_frb` to enable the enhanced fuzzy rule base.
- Uses `DGTSKRegressor` as the underlying model.

### Example

```python
from highfis import DGTSKRegressorEstimator

reg = DGTSKRegressorEstimator(
    n_mfs=3,
    mf_init="kmeans",
    use_en_frb=True,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
reg.fit(X_train, y_train)
r2 = reg.score(X_test, y_test)
```

## LogTSKClassifierEstimator

A scikit-learn compatible classifier wrapper around `LogTSKClassifier`.

### Summary

- Uses `LogTSKClassifier` as the underlying model.
- Implements `fit`, `predict_proba`, `predict`, and `score`.
- `score` returns accuracy.

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

### Summary

- Uses `LogTSKRegressor` as the underlying model.
- Implements `fit`, `predict`, and `score`.
- `score` returns $R^2$.

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

## Exported helper

### `_build_kmeans_input_mfs`

A helper function exported by the module for building Gaussian MFs from
k-means cluster centers and per-feature spread.
