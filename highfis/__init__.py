"""highFIS — sklearn-compatible TSK fuzzy inference system estimators.

This package exposes **Takagi-Sugeno-Kang (TSK) estimators** that follow the
scikit-learn ``fit`` / ``predict`` / ``score`` interface and integrate with
``Pipeline``, ``GridSearchCV``, and other sklearn utilities.

Quick start
-----------
>>> from highfis import HTSKClassifier
>>> clf = HTSKClassifier(n_mfs=5)
>>> clf.fit(X_train, y_train)
>>> clf.predict(X_test)
>>> clf.score(X_test, y_test)

>>> from highfis import HTSKRegressor
>>> reg = HTSKRegressor(n_mfs=5)
>>> reg.fit(X_train, y_train)
>>> reg.predict(X_test)

Estimator families
------------------
Each family comes in a ``*Classifier`` and ``*Regressor`` variant.

- **TSK / HTSK** — vanilla TSK and high-dimensional TSK with geometric mean
  aggregation and log-space softmax normalization.
  ``TSKClassifier``, ``TSKRegressor``, ``HTSKClassifier``, ``HTSKRegressor``.

- **ADATSK** — adaptive softmin antecedent.
  ``ADATSKClassifier``, ``ADATSKRegressor``.

- **ADPTSK** — adaptive double-parameter softmin antecedent.
  ``ADPTSKClassifier``, ``ADPTSKRegressor``.

- **ADMTSK / DombiTSK** — adaptive and fixed-parameter Dombi T-norm antecedent.
  ``ADMTSKClassifier``, ``ADMTSKRegressor``,
  ``DombiTSKClassifier``, ``DombiTSKRegressor``.

- **DGTSK** — double groups of gates for simultaneous feature selection and
  rule extraction.
  ``DGTSKClassifier``, ``DGTSKRegressor``.

- **DGALETSK** — adaptive Ln-Exp softmin with simultaneous feature selection
  and rule extraction.
  ``DGALETSKClassifier``, ``DGALETSKRegressor``.

- **FSREADATSK** — feature selection and rule extraction over ADATSK.
  ``FSREADATSKClassifier``, ``FSREADATSKRegressor``.

- **HDFIS** — high-dimensional inference via minimum or product T-norm.
  ``HDFISMinClassifier``, ``HDFISMinRegressor``,
  ``HDFISProdClassifier``, ``HDFISProdRegressor``.

- **LogTSK** — inverse-log normalization of log-domain rule weights.
  ``LogTSKClassifier``, ``LogTSKRegressor``.

- **MHTSK** — multihead TSK with sparse subantecedents.
  ``MHTSKClassifier``, ``MHTSKRegressor``.

- **AYATSK** — adaptive Yager T-norm antecedent.
  ``AYATSKClassifier``, ``AYATSKRegressor``.

Common parameters
-----------------
n_mfs : int
    Number of membership functions per input feature.  The total number
    of rules depends on ``n_mfs`` and the ``rule_base`` strategy (e.g.
    ``"coco"`` produces exactly ``n_mfs`` rules; ``"cartesian"`` produces
    ``n_mfs ** n_features`` rules).  Ignored when ``input_mfs`` is
    supplied.
mf_init : {"kmeans", "minibatch_kmeans", "fcm", "grid"} or clustering instance
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
    ``n_mfs`` is ignored and ``mf_init`` is skipped; the number of rules
    is inferred from the MF list lengths.  Import MF classes from
    ``highfis.memberships``.
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

To report the versions and runtime settings in use::

    import highfis

    highfis.show_versions()
"""

from ._show_versions import show_versions
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
    MFCacheInfo,
    MHTSKClassifier,
    MHTSKRegressor,
    TSKClassifier,
    TSKRegressor,
    clear_mf_cache,
    mf_cache_info,
    set_mf_cache_enabled,
    set_mf_cache_size,
)
from .optim import BaseTrainer, DGTrainer, GradientTrainer
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
    "BaseTrainer",
    "DGALETSKClassifier",
    "DGALETSKRegressor",
    "DGTSKClassifier",
    "DGTSKRegressor",
    "DGTrainer",
    "DombiTSKClassifier",
    "DombiTSKRegressor",
    "FSREADATSKClassifier",
    "FSREADATSKRegressor",
    "GradientTrainer",
    "HDFISMinClassifier",
    "HDFISMinRegressor",
    "HDFISProdClassifier",
    "HDFISProdRegressor",
    "HTSKClassifier",
    "HTSKRegressor",
    "InputConfig",
    "LogTSKClassifier",
    "LogTSKRegressor",
    "MFCacheInfo",
    "MHTSKClassifier",
    "MHTSKRegressor",
    "TSKClassifier",
    "TSKRegressor",
    "__version__",
    "clear_mf_cache",
    "mf_cache_info",
    "set_mf_cache_enabled",
    "set_mf_cache_size",
    "show_versions",
]
