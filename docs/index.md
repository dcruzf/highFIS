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
- Includes HDFIS variants for product-DMF and minimum frozen-antecedent
  high-dimensional inference.
- Ships with both model-level classes and sklearn-style estimator wrappers.
- Works seamlessly with `Pipeline`, `GridSearchCV`, and standard
  scikit-learn workflows.

## High-level overview

highFIS models combine:

- differentiable membership functions for antecedent fuzzification,
- configurable rule bases and T-norm aggregation,
- normalized rule weights via defuzzification,
- built-in metrics and evaluation utilities for regression and classification,
- task-specific consequent layers for classification or regression.

Use `BaseTSK` for custom pipelines, or choose a concrete model variant when
you want a ready-to-use TSK architecture.

## Quick Start

```bash
pip install highFIS
```

```python
from highfis import HTSKClassifierEstimator

clf = HTSKClassifierEstimator(
    n_mfs=4,
    mf_init="kmeans",
    epochs=150,
    learning_rate=1e-3,
    random_state=42,
)
clf.fit(X_train, y_train)
print(f"Test accuracy: {clf.score(X_test, y_test):.4f}")
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
- [`ADMTSK`](models/admtsk.md) — adaptive Dombi TSK with Composite GMF and
  positive lower-bound membership values.
- [`AYATSK`](models/ayatsk.md) — Yager-style aggregation for more flexible
  antecedent behavior.
- [`AdaTSK`](models/adatsk.md) — adaptive softmin aggregation with dynamic
  rule weighting.
- [`ADPTSK`](models/adptsk.md) — adaptive double-parameter softmin aggregation
  with stable normalized rule weights.
- [`FSRE-AdaTSK`](models/fsre-adatsk.md) — gated feature selection and rule
  extraction inside an adaptive inference pipeline.
- [`DG-TSK`](models/dg-tsk.md) — double-gated training for simultaneous
  feature selection and rule extraction.
- [`DG-ALETSK`](models/dg-aletsk.md) — adaptive Ln-Exp softmin with embedded
  feature and rule gates for sparse high-dimensional modeling.
- [`MHTSK`](models/mhtsk.md) — multihead sparse subantecedents for high-dimensional
  rule extraction and scalable TSK learning.

Each model family exposes both classifier and regressor variants.

## Model selection guide

- Choose `TSK` for a baseline vanilla fuzzy model.
- Choose `HTSK` or `LogTSK` for high-dimensional problems where numerical
  stability is critical.
- Choose `DombiTSK` or `AYATSK` when you want more control over antecedent
  aggregation behavior.
- Choose `HDFIS` when you need high-dimensional inference with either a
  dimension-dependent product antecedent or a frozen minimum antecedent.
- Choose `AdaTSK`, `FSRE-AdaTSK`, `DG-TSK`, or `DG-ALETSK` when you need
  adaptive sparsity, feature gating, or rule extraction.

## Documentation

| Topic | Description |
|---|---|
| [Quick Start](#quick-start) | Installation and first model run. |
| [Models](api/models.md) | Model constructors and usage notes. |
| [Estimators](api/estimators.md) | sklearn-compatible estimator reference. |
| [Layers](api/layers.md) | Layer primitives for fuzzy pipelines. |
| [Defuzzifiers](api/defuzzifiers.md) | Normalization strategies. |
| [T-Norms](api/t_norms.md) | Built-in and custom aggregation functions. |
| [Memberships](api/memberships.md) | Membership functions for antecedents. |
| [Metrics](api/metrics.md) | Regression and classification evaluation utilities. |
| [Base TSK](api/base.md) | Unified training loop and shared logic. |
| [Protocols](api/protocols.md) | Structural typing interfaces. |
| [Persistence](api/persistence.md) | Estimator checkpoint serialization and load validation. |
| [Contributing](contributing.md) | Development setup and contribution guide. |

## Get Started

Use the top-level `highfis` classes for fast prototyping, or extend
`BaseTSK` directly for custom fuzzy pipelines.
