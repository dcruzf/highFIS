# highFIS

[![CI](https://github.com/dcruzf/highFIS/actions/workflows/ci.yaml/badge.svg)](https://github.com/dcruzf/highFIS/actions/workflows/ci.yaml)
[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-green)](https://www.gnu.org/licenses/gpl-3.0)


Python library for high-dimensional Takagi–Sugeno–Kang (TSK) fuzzy inference systems, built on PyTorch with a scikit-learn compatible API.

## Installation

```bash
pip install highFIS
```

**Requirements:** Python ≥ 3.11, PyTorch ≥ 2.3, NumPy ≥ 1.23, scikit-learn ≥ 1.7.

## Quick Start

```python
from highfis import HTSKClassifierEstimator

clf = HTSKClassifierEstimator(
    n_mfs=3,
    rule_base="en",
    epochs=200,
    learning_rate=1e-3,
    ur_weight=0.01,
    random_state=42,
)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

Works with `sklearn.pipeline.Pipeline`, `GridSearchCV`, and `cross_val_score`.

## Overview

Classical TSK systems use a Cartesian product rule base that scales as $s^d$ with input dimensionality $d$, making them intractable for high-dimensional data. highFIS addresses this with:

- **HTSK defuzzification** — geometric-mean firing strengths that remain stable regardless of dimensionality.
- **Compact rule bases** — `"coco"` ($s$ rules) and `"en"` ($s(2d+1)$ rules) for linear scaling.
- **End-to-end differentiability** — all parameters trained jointly via backpropagation.
- **CrossEntropyLoss** — default loss on raw logits (no one-hot encoding), following the PyTSK reference.
- **AdamW optimizer** — separate weight-decay groups: 0 for antecedent (centres/sigmas), configurable for consequent parameters.
- **Early stopping by accuracy** — validation accuracy monitoring with best-model restore.
- **Numerically stable normalization** — `softmax(log(w))` instead of naive division.

## Key Components

| Class | Module | Description |
|---|---|---|
| `GaussianMF` | `highfis.memberships` | Differentiable Gaussian membership function. |
| `MembershipLayer` | `highfis.layers` | Evaluates all membership functions. |
| `RuleLayer` | `highfis.layers` | Computes firing strengths with configurable t-norm and rule base. |
| `NormalizationLayer` | `highfis.layers` | Normalizes firing strengths. |
| `ClassificationConsequentLayer` | `highfis.layers` | Linear TSK consequent aggregation. |
| `HTSKClassifier` | `highfis.models` | Full TSK pipeline as `nn.Module`. |
| `HTSKClassifierEstimator` | `highfis.estimators` | sklearn-compatible estimator. |
| `InputConfig` | `highfis.estimators` | Per-feature membership function configuration. |

## Documentation

Full documentation at [dcruzf.github.io/highFIS](https://dcruzf.github.io/highFIS/).



## License

Distributed under the [GPLv3](LICENSE).
