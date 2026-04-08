---
icon: simple/python
---

# Quick Start

This page provides a minimal setup and first training run using the highFIS sklearn-style estimator.

## Install

```bash
pip install highFIS
```

## Basic Classification Example

```python
from highfis import HTSKClassifierEstimator

clf = HTSKClassifierEstimator(
	n_mfs=3,
	mf_init="kmeans",  # default
	sigma_scale=1.0,
	epochs=200,
	learning_rate=1e-3,
	ur_weight=0.01,
	random_state=42,
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
```

## Using sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from highfis import HTSKClassifierEstimator

pipe = Pipeline([
	("scaler", StandardScaler()),
	("clf", HTSKClassifierEstimator(n_mfs=3, mf_init="kmeans", epochs=150)),
])

pipe.fit(X_train, y_train)
test_acc = pipe.score(X_test, y_test)
```

## Choosing Initialization

- `mf_init="kmeans"` (default): uses k-means centroids as MF centers and
	per-cluster spread for sigma (scaled by `sigma_scale`).
- `mf_init="grid"`: uses per-feature uniform grids controlled by `InputConfig`.

```python
clf = HTSKClassifierEstimator(
		n_mfs=3,
		mf_init="grid",
		rule_base="cartesian",
)
```

## Development Commands

```bash
hatch run typing
hatch run security
hatch test
hatch run docs:serve
```
