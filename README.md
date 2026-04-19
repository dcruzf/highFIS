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
regression workflows.

## 🚀 Quick Start

Install from PyPI:

```bash
pip install highfis
```

Run a classifier:

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

highFIS works with `sklearn.pipeline.Pipeline`, `GridSearchCV`, and
`cross_val_score`.

## 🔧 What’s Included

### Core models

- `HTSKClassifier`, `HTSKRegressor`
- `TSKClassifier`, `TSKRegressor`
- `DombiTSKClassifier`, `DombiTSKRegressor`
- `AdaTSKClassifier`, `AdaTSKRegressor`
- `FSREAdaTSKClassifier`, `FSREAdaTSKRegressor`
- `LogTSKClassifier`, `LogTSKRegressor`

### Estimator wrappers

- `HTSKClassifierEstimator`, `HTSKRegressorEstimator`
- `TSKClassifierEstimator`, `TSKRegressorEstimator`
- `LogTSKClassifierEstimator`, `LogTSKRegressorEstimator`
- `AdaTSKClassifierEstimator`, `AdaTSKRegressorEstimator`
- `FSREAdaTSKClassifierEstimator`, `FSREAdaTSKRegressorEstimator`

### Building blocks

- Memberships: `GaussianMF`, `TriangularMF`, `TrapezoidalMF`, `BellMF`, `SigmoidalMF`, `DiffSigmoidalMF`, `ProdSigmoidalMF`, `SShapedMF`, `LinSShapedMF`, `ZShapedMF`, `LinZShapedMF`, `PiMF`, `GaussianPIMF`
- Defuzzifiers: `SoftmaxLogDefuzzifier`, `SumBasedDefuzzifier`, `LogSumDefuzzifier`
- T-norms: `prod`, `min`, `gmean`, `dombi`
- Rule base strategies: `cartesian`, `coco`, `en`, `custom`

## 📚 Documentation

The published documentation is available at https://dcruzf.github.io/highFIS.

Key reference pages:

- [TSK Vanilla](https://dcruzf.github.io/highFIS/latest/models/tsk-vanilla)
- [LogTSK](https://dcruzf.github.io/highFIS/latest/models/logtsk)
- [HTSK](https://dcruzf.github.io/highFIS/latest/models/htsk)
- [DombiTSK](https://dcruzf.github.io/highFIS/latest/models/dombitsk)
- [AdaTSK](https://dcruzf.github.io/highFIS/latest/models/adatsk)
- [FSRE-AdaTSK](https://dcruzf.github.io/highFIS/latest/models/fsre-adatsk)

## 🤝 Contributing & Development

See the published contribution guide at [contributing](https://dcruzf.github.io/highFIS/latest/contributing/).

## 📄 License

Distributed under the [GPLv3](LICENSE).
