# highFIS

[![CI](https://github.com/dcruzf/highFIS/actions/workflows/ci.yaml/badge.svg)](https://github.com/dcruzf/highFIS/actions/workflows/ci.yaml)
[![Documentation](https://github.com/dcruzf/highFIS/actions/workflows/docs.yml/badge.svg)](https://github.com/dcruzf/highFIS/actions/workflows/docs.yml)
[![DOI](https://img.shields.io/badge/doi-10.5281%2Fzenodo.19489225-%2333CA56?logo=DOI&logoColor=white)](https://doi.org/10.5281/zenodo.19489225)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/highfis)](https://pypi.org/project/highfis/)
[![PyPI - Version](https://img.shields.io/pypi/v/highfis?color=%2333CA56)](https://pypi.org/project/highfis/)
[![PyPI - License](https://img.shields.io/pypi/l/highfis?color=%2333CA56)](https://raw.githubusercontent.com/dcruzf/highFIS/refs/heads/main/LICENSE)



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
| `BaseTSK` | `highfis.base` | Abstract base for TSK models with unified training loop. |
| `GaussianMF` | `highfis.memberships` | Differentiable Gaussian membership function. |
| `TriangularMF` | `highfis.memberships` | Triangular membership function. |
| `TrapezoidalMF` | `highfis.memberships` | Trapezoidal membership function. |
| `BellMF` | `highfis.memberships` | Generalized bell membership function. |
| `SigmoidalMF` | `highfis.memberships` | Sigmoidal membership function. |
| `SoftmaxLogDefuzzifier` | `highfis.defuzzifiers` | Stable `softmax(log(w))` normalization. |
| `SumBasedDefuzzifier` | `highfis.defuzzifiers` | Classic `w / sum(w)` normalization. |
| `LogSumDefuzzifier` | `highfis.defuzzifiers` | Temperature-scaled log-space normalization. |
| `MembershipLayer` | `highfis.layers` | Evaluates all membership functions. |
| `RuleLayer` | `highfis.layers` | Computes firing strengths with configurable t-norm and rule base. |
| `ClassificationConsequentLayer` | `highfis.layers` | Linear TSK consequent aggregation for classification. |
| `RegressionConsequentLayer` | `highfis.layers` | Linear TSK consequent aggregation for regression. |
| `HTSKClassifier` | `highfis.models` | Full TSK classification pipeline as `nn.Module`. |
| `HTSKRegressor` | `highfis.models` | Full TSK regression pipeline as `nn.Module`. |
| `HTSKClassifierEstimator` | `highfis.estimators` | sklearn-compatible classification estimator. |
| `HTSKRegressorEstimator` | `highfis.estimators` | sklearn-compatible regression estimator. |
| `InputConfig` | `highfis.estimators` | Per-feature membership function configuration. |

### Structural Typing Protocols

| Protocol | Description |
|---|---|
| `MembershipFn` | Any callable `(Tensor) → Tensor` for membership degrees. |
| `TNorm` | Any callable `(Tensor) → Tensor` for rule aggregation. |
| `Defuzzifier` | Any callable `(Tensor) → Tensor` for firing-strength normalization. |
| `ConsequentFn` | Any callable `(Tensor, Tensor) → Tensor` for consequent output. |

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
