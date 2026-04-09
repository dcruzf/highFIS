---
icon: lucide/sparkles
---

# highFIS

highFIS is a Python library for high-dimensional Takagi-Sugeno-Kang (TSK) fuzzy systems,
implemented with PyTorch and exposed through a scikit-learn compatible estimator API.

## What Is Included

| Section | Description |
|---|---|
| [Quick Start](markdown.md) | Installation and first training run with `HTSKClassifierEstimator`. |
| [HTSK Technical Notes](htsk-modelo.md) | Mathematical formulation and implementation details of HTSK in highFIS. |
| [Memberships API](api/memberships.md) | Differentiable membership functions (`MembershipFunction`, `GaussianMF`). |
| [T-Norms API](api/t_norms.md) | Built-in aggregators (`prod`, `min`, `gmean`) and custom t-norm injection. |
| [Layers API](api/layers.md) | Membership, rule, normalization, and consequent layers. |
| [Models API](api/models.md) | `HTSKClassifier` end-to-end neural fuzzy model. |
| [Estimators API](api/estimators.md) | `HTSKClassifierEstimator` and `InputConfig` for sklearn workflows. |
| [Contributing](contributing.md) | Development setup, checks, and pull request process. |

## Key Characteristics

- Differentiable fuzzy pipeline end-to-end in PyTorch.
- HTSK inference via geometric-mean firing strengths for high-dimensional stability.
- Estimator default initialization based on k-means (paper-aligned), with grid mode as fallback.
- Rule base strategies: `cartesian`, `coco`, `en`, and `custom`.
- Default loss: `CrossEntropyLoss`; default optimizer: `AdamW` with separate weight-decay groups.
- Early stopping by validation accuracy with automatic best-model restore.
- Numerically stable normalization via `softmax(log(w))`.
- Native integration with `Pipeline`, `GridSearchCV`, and cross-validation.

## Installation

```bash
pip install highFIS
```

Minimum requirements:

- Python 3.10+
- PyTorch 2.3+
- NumPy 1.23+
- scikit-learn 1.7.2+
