"""highFIS — sklearn-compatible TSK fuzzy inference system estimators.

This package exposes **Takagi-Sugeno-Kang (TSK) estimators** that follow the
scikit-learn ``fit`` / ``predict`` / ``score`` interface and integrate with
``Pipeline``, ``GridSearchCV``, and other sklearn utilities.

Quick start
-----------
>>> from highfis import HTSKClassifier
>>> clf = HTSKClassifier(n_rules=10)
>>> clf.fit(X_train, y_train)
>>> clf.predict(X_test)
>>> clf.score(X_test, y_test)

>>> from highfis import HTSKRegressor
>>> reg = HTSKRegressor(n_rules=8)
>>> reg.fit(X_train, y_train)
>>> reg.predict(X_test)

Estimator families
------------------
Each family comes in a ``*Classifier`` and ``*Regressor`` variant.

- **TSK / HTSK** — baseline TSK and hierarchical-softmax activation.
  ``TSKClassifier``, ``TSKRegressor``, ``HTSKClassifier``, ``HTSKRegressor``.

- **ADATSK** — adaptive softmin antecedent (Leite et al., 2022).
  ``ADATSKClassifier``, ``ADATSKRegressor``.

- **ADPTSK** — adaptive Pareto T-norm antecedent.
  ``ADPTSKClassifier``, ``ADPTSKRegressor``.

- **ADMTSK / DombiTSK** — Dombi T-norm antecedent.
  ``ADMTSKClassifier``, ``ADMTSKRegressor``,
  ``DombiTSKClassifier``, ``DombiTSKRegressor``.

- **DGTSK** — data-driven Gaussian antecedent (De Gregório et al., 2023).
  ``DGTSKClassifier``, ``DGTSKRegressor``.

- **DGALETSK** — dimension-adaptive extension of DGTSK.
  ``DGALETSKClassifier``, ``DGALETSKRegressor``.

- **FSREADATSK** — feature selection and rule extraction over ADATSK.
  ``FSREADATSKClassifier``, ``FSREADATSKRegressor``.

- **HDFIS** — hierarchical defuzzification via minimum or product T-norm.
  ``HDFISMinClassifier``, ``HDFISMinRegressor``,
  ``HDFISProdClassifier``, ``HDFISProdRegressor``.

- **LogTSK** — log-domain consequent.
  ``LogTSKClassifier``, ``LogTSKRegressor``.

- **MHTSK** — multi-hierarchical TSK.
  ``MHTSKClassifier``, ``MHTSKRegressor``.

- **AYATSK** — Yager T-norm antecedent.
  ``AYATSKClassifier``, ``AYATSKRegressor``.

Common parameters
-----------------
n_rules : int
    Number of fuzzy rules.  Ignored when ``input_mfs`` is supplied.
mf_init : {"kmeans", "grid"}
    Strategy for initialising input membership functions.
    ``"kmeans"`` (default) fits Gaussian MF centres from k-means cluster
    centroids computed on the training data.
    ``"grid"`` places MFs on a uniform grid; requires ``input_configs``.
input_configs : list[InputConfig] or None
    Per-feature configuration used when ``mf_init="grid"``.  Each
    ``InputConfig`` specifies the feature ``name``, number of MFs
    (``n_mfs``), spacing (``overlap``), and range padding (``margin``).
input_mfs : dict[str, list[MembershipFunction]] or None
    Pre-built membership functions keyed by feature name.  When supplied,
    ``n_rules`` is inferred from the MF list length and ``mf_init`` is
    ignored.  Import MF classes from ``highfis.memberships``.
random_state : int or None
    Seed for reproducible k-means initialisation.

Advanced usage
--------------
For custom membership functions import them explicitly::

    from highfis.memberships import GaussianMF, GaussianPiMF, TrapezoidalMF

For evaluation metrics (beyond sklearn's ``score``)::

    from highfis.metrics import compute_metrics

For direct access to the underlying PyTorch models::

    from highfis.models import HTSKClassifierModel
"""

from .estimators import (
    ADATSKClassifier,
    ADATSKRegressor,
    ADMTSKClassifier,
    ADMTSKRegressor,
    ADPTSKClassifier,
    ADPTSKRegressor,
    AYATSKClassifier,
    AYATSKRegressor,
    DGALETSKClassifier,
    DGALETSKRegressor,
    DGTSKClassifier,
    DGTSKRegressor,
    DombiTSKClassifier,
    DombiTSKRegressor,
    FSREADATSKClassifier,
    FSREADATSKRegressor,
    HDFISMinClassifier,
    HDFISMinRegressor,
    HDFISProdClassifier,
    HDFISProdRegressor,
    HTSKClassifier,
    HTSKRegressor,
    InputConfig,
    LogTSKClassifier,
    LogTSKRegressor,
    MHTSKClassifier,
    MHTSKRegressor,
    TSKClassifier,
    TSKRegressor,
    feature_coverage_rate,
)
from .version import __version__

__all__: list[str] = [
    "ADATSKClassifier",
    "ADATSKRegressor",
    "ADMTSKClassifier",
    "ADMTSKRegressor",
    "ADPTSKClassifier",
    "ADPTSKRegressor",
    "AYATSKClassifier",
    "AYATSKRegressor",
    "DGALETSKClassifier",
    "DGALETSKRegressor",
    "DGTSKClassifier",
    "DGTSKRegressor",
    "DombiTSKClassifier",
    "DombiTSKRegressor",
    "FSREADATSKClassifier",
    "FSREADATSKRegressor",
    "HDFISMinClassifier",
    "HDFISMinRegressor",
    "HDFISProdClassifier",
    "HDFISProdRegressor",
    "HTSKClassifier",
    "HTSKRegressor",
    "InputConfig",
    "LogTSKClassifier",
    "LogTSKRegressor",
    "MHTSKClassifier",
    "MHTSKRegressor",
    "TSKClassifier",
    "TSKRegressor",
    "__version__",
    "feature_coverage_rate",
]
