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
- Ships with both model-level and estimator-level APIs.
- Works seamlessly with `Pipeline`, `GridSearchCV`, and standard
  scikit-learn workflows.

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

highFIS offers a family of TSK fuzzy models optimized for different
high-dimensional behaviors.

- [`Vanilla TSK`](models/tsk-vanilla.md) — the original Takagi-Sugeno-Kang model
  with Gaussian MFs, product antecedent aggregation, and sum-based
  normalization.
- [`HTSK`](models/htsk.md) — high-dimensional TSK with geometric mean
  aggregation and log-space normalization to reduce dimensionality bias.
- [`AYATSK`](models/ayatsk.md) — adaptive Yager aggregation with sum-based
  consequent normalization.
- [`LogTSK`](models/logtsk.md) — inverse-log normalization of log-domain
  rule weights for numerically stable high-dimensional aggregation.
- [`DombiTSK`](models/dombitsk.md) — Dombi t-norm aggregation with first-order
  consequents and a learnable shape parameter.
- [`AdaTSK`](models/adatsk.md) — adaptive Dombi inference using Composite
  Gaussian MFs with a positive lower bound.
- [`FSRE-AdaTSK`](models/fsre-adatsk.md) — adaptive model with gated feature
  selection and rule extraction in the consequent.
- [`DG-TSK`](models/dg-tsk.md) — double-gated training for simultaneous
  feature selection and rule extraction, followed by first-order fine tuning.
- [`DG-ALETSK`](models/dg-aletsk.md) — DG-based adaptive Ln-Exp softmin with
  embedded feature and rule gates for sparse high-dimensional modeling.
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
| [Base TSK](api/base.md) | Unified training loop and shared logic. |
| [Protocols](api/protocols.md) | Structural typing interfaces. |
| [Persistence](api/persistence.md) | Estimator checkpoint serialization and load validation. |
| [Contributing](contributing.md) | Development setup and contribution guide. |

## Get Started

Use the top-level `highfis` classes for fast prototyping, or extend
`BaseTSK` directly for custom fuzzy pipelines.
