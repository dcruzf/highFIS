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
| [Protocols API](api/protocols.md) | Structural typing protocols (`MembershipFn`, `TNorm`, `Defuzzifier`, `ConsequentFn`). |
| [Memberships API](api/memberships.md) | Differentiable membership functions (`GaussianMF`, `TriangularMF`, `TrapezoidalMF`, `BellMF`, `SigmoidalMF`). |
| [T-Norms API](api/t_norms.md) | Built-in aggregators (`prod`, `min`, `gmean`) and custom t-norm injection. |
| [Defuzzifiers API](api/defuzzifiers.md) | Pluggable firing-strength normalization (`SoftmaxLogDefuzzifier`, `SumBasedDefuzzifier`, `LogSumDefuzzifier`). |
| [Layers API](api/layers.md) | Membership, rule, and consequent layers. |
| [Base TSK API](api/base.md) | `BaseTSK` abstract base with unified training loop. |
| [Models API](api/models.md) | `HTSKClassifier` and `HTSKRegressor` end-to-end neural fuzzy models. |
| [Estimators API](api/estimators.md) | `HTSKClassifierEstimator`, `HTSKRegressorEstimator`, and `InputConfig` for sklearn workflows. |
| [Contributing](contributing.md) | Development setup, checks, and pull request process. |

## Key Characteristics

- Differentiable fuzzy pipeline end-to-end in PyTorch.
- `BaseTSK` abstract base with unified training loop — extend to create custom models.
- Structural typing protocols for all pipeline stages.
- Five membership function types: Gaussian, Triangular, Trapezoidal, Bell, Sigmoidal.
- Pluggable defuzzifiers: `SoftmaxLogDefuzzifier` (default), `SumBasedDefuzzifier`, `LogSumDefuzzifier`.
- HTSK inference via geometric-mean firing strengths for high-dimensional stability.
- Estimator default initialization based on k-means (paper-aligned), with grid mode as fallback.
- Rule base strategies: `cartesian`, `coco`, `en`, and `custom`.
- Default loss: `CrossEntropyLoss` (classifier) / `MSELoss` (regressor); default optimizer: `AdamW` with separate weight-decay groups.
- Early stopping with automatic best-model restore.
- Native integration with `Pipeline`, `GridSearchCV`, and cross-validation.

## Installation

```bash
pip install highFIS
```

Minimum requirements:

- Python 3.11+
- PyTorch 2.3+
- NumPy 1.23+
- scikit-learn 1.7.2+
