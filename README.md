# highFIS

[![CI](https://github.com/dcruzf/highFIS/actions/workflows/ci.yaml/badge.svg)](https://github.com/dcruzf/highFIS/actions/workflows/ci.yaml)
[![Documentation](https://github.com/dcruzf/highFIS/actions/workflows/docs.yml/badge.svg)](https://github.com/dcruzf/highFIS/actions/workflows/docs.yml)
[![DOI](https://img.shields.io/badge/doi-10.5281%2Fzenodo.19489225-%2333CA56?logo=DOI&logoColor=white)](https://doi.org/10.5281/zenodo.19489225)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/highfis)](https://pypi.org/project/highfis/)
[![PyPI - Version](https://img.shields.io/pypi/v/highfis?color=%2333CA56)](https://pypi.org/project/highfis/)
[![PyPI - License](https://img.shields.io/pypi/l/highfis?color=%2333CA56)](https://raw.githubusercontent.com/dcruzf/highFIS/refs/heads/main/LICENSE)

highFIS is a PyTorch-based framework for high-dimensional Takagi–Sugeno–Kang
(TSK) fuzzy systems. It brings differentiable fuzzy inference, numerical
stability, and sklearn-compatible estimators to both classification and
regression. The library also includes DGTSK dynamic-gating models for feature
and rule selection in high-dimensional fuzzy systems.

## 🚀 Overview

- Differentiable TSK fuzzy systems built for high-dimensional data.
- Supports both concrete model classes and sklearn-style estimator wrappers.
- Includes adaptive and gated inference variants for feature selection and
  sparse rule extraction.
- Designed for numerical stability with log-space and inverse-log defuzzifiers.

## 📦 Installation

Install from PyPI:

```bash
pip install highfis
```

## 🧠 Quick Start

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

highFIS integrates with `sklearn.pipeline.Pipeline`, `GridSearchCV`, and
`cross_val_score`.

## 🧩 Model families

highFIS provides a full family of TSK models, each tuned for a specific
high-dimensional inference strategy.

- `TSK` — vanilla TSK with product antecedent aggregation and sum-based
  normalization.
- `HTSK` — high-dimensional TSK with geometric mean aggregation and log-space
  normalization.
- `LogTSK` — log-domain inverse-log normalization for stable aggregation.
- `DombiTSK` — Dombi t-norm aggregation with a learnable shape parameter.
- `ADMTSK` — adaptive Dombi TSK with Composite Gaussian membership functions.
- `AYATSK` — Yager aggregation for flexible antecedent behavior.
- `AdaTSK` — adaptive softmin-style inference with dynamic rule weighting.
- `FSRE-AdaTSK` — adaptive model with gated feature selection and rule extraction.
- `DG-TSK` — double-gated training for feature selection and rule extraction.
- `DG-ALETSK` — adaptive Ln-Exp softmin with embedded feature and rule gates.
- `HDFIS` — high-dimensional inference with product DMF (`HDFIS-prod`) and minimum T-norm frozen antecedents (`HDFIS-min`).

Each family exposes classifier and regressor variants.

## 🔧 Core components

### Models

- `HTSKClassifier`, `HTSKRegressor`
- `TSKClassifier`, `TSKRegressor`
- `LogTSKClassifier`, `LogTSKRegressor`
- `DombiTSKClassifier`, `DombiTSKRegressor`
- `ADMTSKClassifier`, `ADMTSKRegressor`
- `AYATSKClassifier`, `AYATSKRegressor`
- `AdaTSKClassifier`, `AdaTSKRegressor`
- `FSREAdaTSKClassifier`, `FSREAdaTSKRegressor`
- `DGTSKClassifier`, `DGTSKRegressor`
- `DGALETSKClassifier`, `DGALETSKRegressor`
- `HDFISProdClassifier`, `HDFISProdRegressor`
- `HDFISMinClassifier`, `HDFISMinRegressor`

### Estimator wrappers

- `HTSKClassifierEstimator`, `HTSKRegressorEstimator`
- `TSKClassifierEstimator`, `TSKRegressorEstimator`
- `LogTSKClassifierEstimator`, `LogTSKRegressorEstimator`
- `DombiTSKClassifierEstimator`, `DombiTSKRegressorEstimator`
- `ADMTSKClassifierEstimator`, `ADMTSKRegressorEstimator`
- `DGALETSKClassifierEstimator`, `DGALETSKRegressorEstimator`
- `DGTSKClassifierEstimator`, `DGTSKRegressorEstimator`
- `FSREAdaTSKClassifierEstimator`, `FSREAdaTSKRegressorEstimator`
- `AYATSKClassifierEstimator`, `AYATSKRegressorEstimator`
- `AdaTSKClassifierEstimator`, `AdaTSKRegressorEstimator`
- `HDFISProdClassifierEstimator`, `HDFISProdRegressorEstimator`
- `HDFISMinClassifierEstimator`, `HDFISMinRegressorEstimator`

### Building blocks

- Memberships: `GaussianMF`, `TriangularMF`, `TrapezoidalMF`, `BellMF`,
  `SigmoidalMF`, `DiffSigmoidalMF`, `ProdSigmoidalMF`, `SShapedMF`,
  `LinSShapedMF`, `ZShapedMF`, `LinZShapedMF`, `PiMF`,
  `CompositeExponentialMF`, `GaussianPIMF`
- Defuzzifiers: `SoftmaxLogDefuzzifier`, `SumBasedDefuzzifier`,
  `LogSumDefuzzifier`
- T-norms: `prod`, `min`, `gmean`, `dombi`, `yager`, `yager_simple`,
  `ale_softmin_yager`
- Rule base strategies: `cartesian`, `coco`, `en`, `custom`
- Persistence: checkpoint save/load and validation via `highfis.persistence`

## 🛠️ Training options

highFIS uses gradient-based optimization and supports:

- adaptive optimizers like Adam/W and standard SGD
- early stopping with validation
- uniform rule regularization for balanced rule activation
- custom T-norms, custom rule bases, and custom defuzzifiers

## 📚 Documentation

The published documentation is available at:

https://dcruzf.github.io/highFIS

Key reference pages:

- [TSK Vanilla](https://dcruzf.github.io/highFIS/latest/models/tsk-vanilla)
- [LogTSK](https://dcruzf.github.io/highFIS/latest/models/logtsk)
- [AYATSK](https://dcruzf.github.io/highFIS/latest/models/ayatsk)
- [HTSK](https://dcruzf.github.io/highFIS/latest/models/htsk)
- [DombiTSK](https://dcruzf.github.io/highFIS/latest/models/dombitsk)
- [ADMTSK](https://dcruzf.github.io/highFIS/latest/models/admtsk)
- [AdaTSK](https://dcruzf.github.io/highFIS/latest/models/adatsk)
- [DGTSK](https://dcruzf.github.io/highFIS/latest/models/dg-tsk)
- [DG-ALETSK](https://dcruzf.github.io/highFIS/latest/models/dg-aletsk)
- [FSRE-AdaTSK](https://dcruzf.github.io/highFIS/latest/models/fsre-adatsk)
- [HDFIS](https://dcruzf.github.io/highFIS/latest/models/hdfis)

## 🧪 Testing & quality

Run the test suite with coverage:

```bash
hatch test -c --a
```

Format and lint the repository:

```bash
hatch fmt
```

Run static type checks:

```bash
hatch run typing
```

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests, and refer to
our development guide in the documentation:

https://dcruzf.github.io/highFIS/latest/contributing/

## 📄 License

Distributed under the [GPLv3](LICENSE).
