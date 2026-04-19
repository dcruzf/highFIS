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

### Classification

- `HTSKClassifier` — geometric-mean inference for high-dimensional stability.
- `TSKClassifier` — classical product t-norm TSK.
- `DombiTSKClassifier` — Dombi aggregation with a tunable shape parameter.
- `AdaTSKClassifier` — adaptive Dombi inference with per-rule learnable shape.
- `FSREAdaTSKClassifier` — adaptive softmin with feature selection and rule
  extraction.
- `LogTSKClassifier` — log-space normalization with temperature control.

### Regression

- `HTSKRegressor`
- `TSKRegressor`
- `DombiTSKRegressor`
- `AdaTSKRegressor`
- `FSREAdaTSKRegressor`
- `LogTSKRegressor`

## Documentation

| Topic | Description |
|---|---|
| [Quick Start](#quick-start) | Installation and first model run. |
| [HTSK Technical Notes](htsk-modelo.md) | Deep dive into HTSK inference design. |
| [TSK Vanilla](models/tsk-vanilla.md) | Standard Takagi-Sugeno-Kang model. |
| [LogTSK](models/logtsk.md) | Log-space TSK for stability in high dimensions. |
| [AdaTSK](models/adatsk.md) | Adaptive Dombi TSK with learned shape. |
| [FSRE-AdaTSK](models/fsre-adatsk.md) | Feature selection and rule extraction. |
| [Models API](api/models.md) | Model constructors and usage notes. |
| [Estimators API](api/estimators.md) | sklearn-compatible estimator reference. |
| [Layers API](api/layers.md) | Layer primitives for fuzzy pipelines. |
| [Defuzzifiers API](api/defuzzifiers.md) | Normalization strategies. |
| [T-Norms API](api/t_norms.md) | Built-in and custom aggregation functions. |
| [Memberships API](api/memberships.md) | Membership functions for antecedents. |
| [Base TSK API](api/base.md) | Unified training loop and shared logic. |
| [Protocols API](api/protocols.md) | Structural typing interfaces. |
| [Contributing](contributing.md) | Development setup and contribution guide. |

## Get Started

Use the top-level `highfis` classes for fast prototyping, or extend
`BaseTSK` directly for custom fuzzy pipelines.
