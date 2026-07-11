---
icon: lucide/sparkles
---

# highFIS

highFIS is a modern PyTorch library for high-dimensional Takagi-Sugeno-Kang
(TSK) fuzzy systems. It delivers differentiable, trainable fuzzy inference for
classification and regression, with sklearn-compatible estimators for fast
experimentation.

## Why highFIS?

- Built for high-dimensional data and numerical stability.
- Supports adaptive and gated fuzzy inference, including feature selection
  and rule extraction.
- Includes HDFIS variants for product T-norm and minimum T-norm
  high-dimensional inference.
- Ships sklearn-compatible estimators (`*Classifier` / `*Regressor`) for every
  model family.
- Works seamlessly with `Pipeline`, `GridSearchCV`, and standard
  scikit-learn workflows.

## High-level overview

highFIS models combine:

- differentiable membership functions for antecedent fuzzification,
- configurable rule bases and T-norm aggregation,
- normalized rule weights via defuzzification,
- built-in metrics and evaluation utilities for regression and classification,
- task-specific consequent layers for classification or regression.

Use a model class from `highfis.models` for custom PyTorch pipelines, or
import an estimator directly from `highfis` for sklearn-compatible training.

## Quick Start

```bash
pip install highfis
```

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from highfis import HTSKClassifier

# 1. Generate synthetic data
X, y = make_classification(n_samples=800, n_features=10, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Scale features (essential for fuzzy membership functions)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Fit the high-dimensional HTSK classifier
clf = HTSKClassifier(
    n_mfs=3,                  # Number of clusters (3 fuzzy rules)
    mf_init="kmeans",         # Initialise MFs using K-Means clustering
    epochs=100,
    learning_rate=0.01,
    random_state=42,
)
clf.fit(X_train_scaled, y_train)

# 4. Evaluate performance
test_accuracy = clf.score(X_test_scaled, y_test)
print(f"Test accuracy: {test_accuracy:.2%}")
```

## Models Available

highFIS includes the following concrete TSK model families:

- [`TSK`](models/tsk-vanilla.md) — vanilla TSK with product antecedent
  aggregation and sum-based normalization.
- [`HTSK`](models/htsk.md) — high-dimensional TSK with geometric mean
  aggregation and log-space softmax normalization.
- [`LogTSK`](models/logtsk.md) — inverse-log normalization of log-domain
  rule weights for stable high-dimensional inference.
- [`HDFIS`](models/hdfis.md) — high-dimensional inference with both
  product-DMF aggregation (HDFIS-prod) and minimum frozen-antecedent
  inference (HDFIS-min).
- [`DombiTSK`](models/dombitsk.md) — Dombi parametric aggregation with a
  learnable shape parameter.
- [`ADMTSK`](models/admtsk.md) — adaptive Dombi TSK with dimension-dependent
  Gaussian membership functions.
- [`AYATSK`](models/ayatsk.md) — Yager-style aggregation for more flexible
  antecedent behavior.
- [`ADATSK`](models/adatsk.md) — adaptive softmin aggregation with dynamic
  rule weighting.
- [`ADPTSK`](models/adptsk.md) — adaptive double-parameter softmin aggregation
  with stable normalized rule weights.
- [`FSRE-ADATSK`](models/fsre-adatsk.md) — gated feature selection and rule
  extraction inside an adaptive inference pipeline.
- [`DGTSK`](models/dg-tsk.md) — double-gated training for simultaneous
  feature selection and rule extraction.
- [`DGALETSK`](models/dg-aletsk.md) — adaptive Ln-Exp softmin with embedded
  feature and rule gates for sparse high-dimensional modeling.
- [`MHTSK`](models/mhtsk.md) — multihead sparse subantecedents for high-dimensional
  rule extraction and scalable TSK learning.

Each model family exposes both classifier and regressor variants.

## Documentation

| Topic | Description |
|---|---|
| [Quick Start](#quick-start) | Installation and first model run. |
| [Model Families](models/index.md) | Guide to the 13 available neuro-fuzzy model architectures. |
| [User Guides](guides/optimisers.md) | Guides for optimization, introspection, initialization, and tuning. |
| [Cookbook](cookbook/index.md) | Short, runnable recipes for common tasks. |
| [API Reference](api/index.md) | Complete reference documentation for all public modules. |
| [Estimators](api/estimators.md) | sklearn-compatible estimator reference. |
| [Models](api/models.md) | Model constructors and usage notes. |
| [Layers](api/layers.md) | Layer primitives for fuzzy pipelines. |
| [Defuzzifiers](api/defuzzifiers.md) | Normalization strategies. |
| [T-Norms](api/t_norms.md) | Built-in and custom aggregation functions. |
| [Memberships](api/memberships.md) | Membership functions for antecedents. |
| [Metrics](api/metrics.md) | Regression and classification evaluation utilities. |
| [Base TSK](api/models.md#highfis.models.BaseTSK) | Unified training loop and shared logic. |
| [Protocols](api/protocols.md) | Structural typing interfaces. |
| [Persistence](api/persistence.md) | Estimator checkpoint serialization and load validation. |
| [Contributing](contributing.md) | Development setup and contribution guide. |

## Get Started

Import [estimators](api/estimators.md) directly from `highfis` for sklearn-compatible usage, or access PyTorch [model classes](api/models.md) via `highfis.models` for custom training loops.
