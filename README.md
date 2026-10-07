# highFIS

[![CI](https://github.com/dcruzf/highFIS/actions/workflows/ci.yaml/badge.svg)](https://github.com/dcruzf/highFIS/actions/workflows/ci.yaml)
[![Documentation](https://github.com/dcruzf/highFIS/actions/workflows/docs.yml/badge.svg)](https://github.com/dcruzf/highFIS/actions/workflows/docs.yml)
[![DOI](https://img.shields.io/badge/doi-10.5281%2Fzenodo.19489225-%2333CA56?logo=DOI&logoColor=white)](https://doi.org/10.5281/zenodo.19489225)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/highfis)](https://pypi.org/project/highfis/)
[![PyPI - Version](https://img.shields.io/pypi/v/highfis?color=%2333CA56)](https://pypi.org/project/highfis/)
[![PyPI - License](https://img.shields.io/pypi/l/highfis?color=%2333CA56)](https://raw.githubusercontent.com/dcruzf/highFIS/refs/heads/main/LICENSE)

highFIS is a PyTorch-based library for high-dimensional Takagi–Sugeno–Kang
(TSK) fuzzy systems. It brings differentiable fuzzy inference, numerical
stability, and scikit-learn compatible estimators to both classification and
regression.

## 🚀 Overview

- Differentiable TSK fuzzy systems built for high-dimensional data.
- Thirteen model families from the literature behind one API, each with a
  classifier and a regressor.
- Available as scikit-learn compatible estimators and as plain PyTorch model
  classes.
- Designed for numerical stability with log-space and inverse-log defuzzifiers.

## 📦 Installation

Install from PyPI:

```bash
pip install highfis
```

highFIS requires Python 3.11 or newer and depends on PyTorch, NumPy,
scikit-learn, and tqdm. The diagnostic plots need matplotlib, an optional
dependency:

```bash
pip install highfis[plot]
```

## 🧠 Quick Start

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier

X, y = make_classification(n_samples=800, n_features=10, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Membership functions are sensitive to feature scale.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = HTSKClassifier(
    n_mfs=3,
    mf_init="kmeans",
    epochs=100,
    learning_rate=0.01,
    random_state=42,
)
clf.fit(X_train, y_train)
print(f"Test accuracy: {clf.score(X_test, y_test):.2%}")
```

highFIS integrates with `sklearn.pipeline.Pipeline`, `GridSearchCV`, and
`cross_val_score`.

## 🧩 Model families

highFIS implements thirteen TSK model families, each following a published
high-dimensional inference strategy.

- `TSK` — generic TSK; by default the vanilla system with product antecedent
  aggregation and sum-based normalization, with selectable membership
  function, T-norm, and defuzzifier.
- `HTSK` — geometric mean aggregation with log-space softmax normalization.
- `LogTSK` — inverse-log normalization of log-domain rule weights.
- `HDFIS` — high-dimensional inference with product T-norm (`HDFISProd`) and
  minimum T-norm (`HDFISMin`) variants.
- `DombiTSK` — Dombi T-norm aggregation with a fixed shape parameter.
- `ADMTSK` — adaptive Dombi T-norm, with the shape parameter derived from the
  input dimensionality, and composite Gaussian membership functions.
- `AYATSK` — adaptive Yager T-norm with composite exponential membership
  functions.
- `ADATSK` — adaptive softmin antecedent aggregation.
- `ADPTSK` — adaptive double-parameter softmin antecedent aggregation.
- `FSRE-ADATSK` — ADATSK extended with embedded feature selection and rule
  extraction.
- `DGTSK` — double groups of gates for simultaneous feature selection and rule
  extraction.
- `DGALETSK` — adaptive Ln-Exp softmin with simultaneous feature selection and
  rule extraction.
- `MHTSK` — multihead TSK built from sparse subantecedents over random feature
  subsets.

Each family exposes classifier and regressor variants.

## 🔧 Core components

highFIS exposes a compact, model-family-driven API with both concrete
model classes and sklearn-compatible estimator wrappers.

- Estimators: `*Classifier` and `*Regressor` variants for each model family,
  importable directly from `highfis` (for example `HTSKClassifier`,
  `FSREADATSKRegressor`, `HDFISProdClassifier`)
- PyTorch models: the underlying `*ClassifierModel` and `*RegressorModel`
  classes in `highfis.models`
- Building blocks: membership functions (`highfis.memberships`), defuzzifiers
  (`highfis.defuzzifiers`), T-norms (`highfis.t_norms`), and layers
  (`highfis.layers`)
- Interpretation: `estimator.rules_as_text()` writes the rule base as
  `IF ... THEN ...` sentences; the rule table, consequent coefficients, and
  gate values are available as estimator methods
- Diagnostic plots: `estimator.plot(kind=...)` for the learned membership
  functions, the training history, the rule activations, and the predictions
- Utilities: evaluation metrics (`highfis.metrics`), estimator checkpoints
  (`highfis.persistence`), and a membership-function initialization cache

For the full class list and API reference, see the documentation:

- [Models](https://dcruzf.github.io/highFIS/latest/api/models)
- [Estimators](https://dcruzf.github.io/highFIS/latest/api/estimators)

## 🛠️ Training options

highFIS uses gradient-based optimization and supports:

- an optimizer selected per model family (SGD, Adam, or AdamW), following the
  source paper
- mini-batch training with learning-rate schedulers and weight decay on the
  consequent parameters
- early stopping on a validation set passed to `fit`
- uniform regularization (`ur_weight`) for balanced rule activation
- a choice of membership function, T-norm, and defuzzifier in the generic
  `TSKClassifier` and `TSKRegressor`; the other families keep their published
  combination
- custom T-norms, rule bases, and defuzzifiers through the PyTorch model
  classes

## 📚 Documentation

The published documentation is available at:

https://dcruzf.github.io/highFIS

Start with the [model families](https://dcruzf.github.io/highFIS/latest/models/)
overview, the [user guides](https://dcruzf.github.io/highFIS/latest/guides/optimisers/),
and the [cookbook](https://dcruzf.github.io/highFIS/latest/cookbook/). The
[diagnostic plots](https://dcruzf.github.io/highFIS/latest/guides/plotting/) guide
shows how to inspect a fitted model.

Model reference pages:

- [TSK Vanilla](https://dcruzf.github.io/highFIS/latest/models/tsk-vanilla)
- [LogTSK](https://dcruzf.github.io/highFIS/latest/models/logtsk)
- [AYATSK](https://dcruzf.github.io/highFIS/latest/models/ayatsk)
- [HTSK](https://dcruzf.github.io/highFIS/latest/models/htsk)
- [DombiTSK](https://dcruzf.github.io/highFIS/latest/models/dombitsk)
- [ADPTSK](https://dcruzf.github.io/highFIS/latest/models/adptsk)
- [ADMTSK](https://dcruzf.github.io/highFIS/latest/models/admtsk)
- [ADATSK](https://dcruzf.github.io/highFIS/latest/models/adatsk)
- [DGTSK](https://dcruzf.github.io/highFIS/latest/models/dg-tsk)
- [DG-ALETSK](https://dcruzf.github.io/highFIS/latest/models/dg-aletsk)
- [FSRE-ADATSK](https://dcruzf.github.io/highFIS/latest/models/fsre-adatsk)
- [MHTSK](https://dcruzf.github.io/highFIS/latest/models/mhtsk)
- [HDFIS](https://dcruzf.github.io/highFIS/latest/models/hdfis)

## 🧪 Testing & quality

Format, lint, and type check:

```bash
hatch check --fix
```

Run the test suite with coverage:

```bash
hatch test -c -a
```

Run security scan:

```bash
hatch run security
```

## 📝 Citation

If you use highFIS in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff) or the
[Zenodo DOI](https://doi.org/10.5281/zenodo.19489225).

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests, and refer to
our development guide in the documentation: [contributing](https://dcruzf.github.io/highFIS/latest/contributing/).

When reporting a bug, include the output of `highfis.show_versions()`.

## 📄 License

Distributed under the [GPLv3](LICENSE).
