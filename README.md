# highFIS

[![CI](https://github.com/dcruzf/highFIS/actions/workflows/ci.yaml/badge.svg)](https://github.com/dcruzf/highFIS/actions/workflows/ci.yaml)
[![Documentation](https://github.com/dcruzf/highFIS/actions/workflows/docs.yml/badge.svg)](https://github.com/dcruzf/highFIS/actions/workflows/docs.yml)
[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-green)](https://www.gnu.org/licenses/gpl-3.0)


Python library for high-dimensional Takagi–Sugeno–Kang (TSK) fuzzy inference systems, built on PyTorch with a scikit-learn compatible API.

## 📦 Installation

Install from PyPI:

```bash
pip install highfis
```

## 🧠 Quick Start

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

## 🧩 Key Components

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

## 🧪 Testing & Quality

### Running tests

Run the full test suite with coverage:

```bash
hatch test -c -a
```

This project is tested on Python 3.11 | 3.12 | 3.13 | 3.14 across Linux, Windows and macOS.

### Linting & Formatting

```bash
hatch fmt
```

### Typing

```bash
hatch run typing
```

### Security

```bash
hatch run security
```

## 📚 Documentation

Comprehensive guides, API reference, and examples: [dcruzf.github.io/highFIS](https://dcruzf.github.io/highFIS/).

## 🤝 Contributing

Issues and pull requests are welcome! Please open a discussion if you'd like to propose larger changes.

## 📄 License

Distributed under the [GPLv3](LICENSE).
