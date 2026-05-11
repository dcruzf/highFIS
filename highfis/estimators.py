"""Scikit-learn compatible estimator wrappers for highFIS TSK models.

This module provides high-level, sklearn-compatible wrappers for every TSK
variant implemented in `highfis.models`. Each estimator follows the standard
`fit` / `predict` / `score` interface and handles membership-function
initialization, model construction, and the training loop internally.

Base Classes:
    Two abstract base classes share the common logic:

    - `_BaseClassifierEstimator`: For classification tasks.
    - `_BaseRegressorEstimator`: For regression tasks.

Model Family Overview:
    Concrete estimators cover the following model families:

    **TSK**
        Vanilla Takagi-Sugeno-Kang model.

        Implemented by:
            `TSKClassifierEstimator`, `TSKRegressorEstimator`

    **HTSK**
        High-dimensional TSK via averaged defuzzification.

        Implemented by:
            `HTSKClassifierEstimator`, `HTSKRegressorEstimator`

    **LogTSK**
        Inverse-log normalization of log-domain rule weights for
        high-dimensional data.

        Implemented by:
            `LogTSKClassifierEstimator`, `LogTSKRegressorEstimator`

    **DombiTSK**
        Dombi T-norm based TSK.

        Implemented by:
            `DombiTSKClassifierEstimator`, `DombiTSKRegressorEstimator`

    **ADMTSK**
        Adaptive Dombi TSK with Composite Gaussian membership functions.

        Implemented by:
            `ADMTSKClassifierEstimator`, `ADMTSKRegressorEstimator`

    **AYATSK**
        Adaptive Yager T-norm based TSK.

        Implemented by:
            `AYATSKClassifierEstimator`, `AYATSKRegressorEstimator`

    **AdaTSK**
        Adaptive softmin based TSK.

        Implemented by:
            `AdaTSKClassifierEstimator`, `AdaTSKRegressorEstimator`

    **FSRE-AdaTSK**
        AdaTSK with feature-selection and rule-extraction gates.

        Implemented by:
            `FSREAdaTSKClassifierEstimator`, `FSREAdaTSKRegressorEstimator`

    **DG-ALETSK**
        Double-gate adaptive Ln-Exp softmin TSK.

        Implemented by:
            `DGALETSKClassifierEstimator`, `DGALETSKRegressorEstimator`

    **DG-TSK**
        Double-gate TSK with point-based FRB.

        Implemented by:
            `DGTSKClassifierEstimator`, `DGTSKRegressorEstimator`

    **HDFIS**
        High-dimensional inference with both product DMF and minimum
        frozen-antecedent variants.

        Implemented by:
            `HDFISProdClassifierEstimator`, `HDFISProdRegressorEstimator`,
            `HDFISMinClassifierEstimator`, `HDFISMinRegressorEstimator`

Membership Function Initialization:
    The following strategies are available for initializing membership functions:

    - `mf_init="kmeans"` (default):
        K-means cluster centroids are used as membership function centers.
        The sigma values are derived from within-cluster spread and scaled
        by `sigma_scale`. This produces a CoCo rule base by default.

    - `mf_init="grid"`:
        Regular grid placement controlled by `InputConfig`. This produces
        a Cartesian rule base by default.

Notes:
    - All estimators follow the scikit-learn API design.
    - Model construction and training are fully encapsulated within the
      estimator interface.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Self, cast

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .base import BaseTSK
from .memberships import CompositeGMF, DimensionDependentGaussianMF, GaussianMF
from .metrics import compute_metrics
from .models import (
    AdaTSKClassifier,
    AdaTSKRegressor,
    ADMTSKClassifier,
    ADMTSKRegressor,
    AYATSKClassifier,
    AYATSKRegressor,
    DGALETSKClassifier,
    DGALETSKRegressor,
    DGTSKClassifier,
    DGTSKRegressor,
    DombiTSKClassifier,
    DombiTSKRegressor,
    FSREAdaTSKClassifier,
    FSREAdaTSKRegressor,
    HDFISMinClassifier,
    HDFISMinRegressor,
    HDFISProdClassifier,
    HDFISProdRegressor,
    HTSKClassifier,
    HTSKRegressor,
    LogTSKClassifier,
    LogTSKRegressor,
    TSKClassifier,
    TSKRegressor,
)
from .persistence import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_payload,
)


@dataclass(frozen=True)
class InputConfig:
    """Per-feature configuration for Gaussian MF grid initialisation.

    This dataclass controls how membership functions are placed on a single
    input feature when ``mf_init="grid"``. When ``mf_init="kmeans"`` only
    the ``name`` field is used; centres and sigmas are derived from k-means
    cluster centroids.

    Attributes:
        name: Feature name. Used as the key in the membership-function
            dictionary passed to the underlying TSK model.
        n_mfs: Number of Gaussian MFs to place on this feature. Must be
            ``>= 1``.
        overlap: Spacing factor between neighbouring MF centres. A larger
            value widens each MF (more overlap); ``0.5`` corresponds to
            roughly half-width overlap at the midpoint between centres.
        margin: Fractional padding added to the observed feature range before
            centre placement. ``0.10`` extends each side of ``[x_min, x_max]``
            by 10 percent so edge centres are not clipped to extreme values.

    Example:
        ```python
        from highfis.estimators import InputConfig

        configs = [
            InputConfig(name="sepal_length", n_mfs=3),
            InputConfig(name="sepal_width", n_mfs=5, overlap=0.3),
        ]
        ```
    """

    name: str
    n_mfs: int = 3
    overlap: float = 0.5
    margin: float = 0.10


def _build_gaussian_input_mfs(
    x: np.ndarray,
    input_configs: list[InputConfig],
) -> dict[str, list[GaussianMF]]:
    """Build Gaussian MFs per input using grid initialization."""
    input_mfs: dict[str, list[GaussianMF]] = {}
    for idx, cfg in enumerate(input_configs):
        if cfg.n_mfs < 1:
            raise ValueError(f"n_mfs for '{cfg.name}' must be >= 1")

        x_col = x[:, idx]
        x_min = float(np.min(x_col))
        x_max = float(np.max(x_col))
        pad = (x_max - x_min) * float(cfg.margin)
        rmin = x_min - pad
        rmax = x_max + pad
        if rmax <= rmin:
            rmax = rmin + 1e-3

        centers = np.linspace(rmin, rmax, int(cfg.n_mfs), dtype=np.float64)
        if cfg.n_mfs == 1:
            width = max((rmax - rmin), 1e-3)
        else:
            spacing = (rmax - rmin) / float(cfg.n_mfs - 1)
            width = max(spacing * (1.0 + float(cfg.overlap)), 1e-3)
        sigma = max(width / 2.0, 1e-3)

        input_mfs[cfg.name] = [GaussianMF(mean=float(c), sigma=float(sigma)) for c in centers]

    return input_mfs


def _build_kmeans_input_mfs(
    x: np.ndarray,
    n_clusters: int,
    sigma_scale: float,
    feature_names: list[str],
    random_state: int | None,
) -> dict[str, list[GaussianMF]]:
    r"""Build Gaussian MFs via k-means cluster-center initialization.

    Follows Cui et al. (IJCNN 2021): the center of MF (r, d) is set to the
    d-th coordinate of the r-th k-means centroid.  The initial sigma is
    sampled from :math:`\\mathcal{N}(h, 0.2)` where *h* equals
    *sigma_scale* multiplied by the within-cluster standard deviation of
    feature *d* in cluster *r*.  When a cluster has near-zero spread, the
    base sigma falls back to half the gap to the nearest neighbouring
    centroid in that feature dimension.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(x)
    centers: np.ndarray = km.cluster_centers_  # (n_clusters, n_features)
    labels: np.ndarray = np.asarray(km.labels_)
    rng = np.random.default_rng(random_state)

    input_mfs: dict[str, list[GaussianMF]] = {}
    for d, name in enumerate(feature_names):
        col = x[:, d]
        center_col = centers[:, d]
        mfs: list[GaussianMF] = []
        for r in range(n_clusters):
            c = float(center_col[r])
            mask = labels == r
            raw_sigma = float(np.std(col[mask])) if int(mask.sum()) > 1 else 0.0
            if raw_sigma < 1e-6:
                # Half the minimum gap to a neighbouring centroid in this dimension
                other = np.delete(center_col, r)
                raw_sigma = float(np.min(np.abs(other - c))) / 2.0 if len(other) > 0 else 1.0
            h = raw_sigma * sigma_scale
            sigma = max(float(rng.normal(loc=h, scale=0.2)), 1e-3)
            mfs.append(GaussianMF(mean=c, sigma=sigma))
        input_mfs[name] = mfs

    return input_mfs


def _build_pfrb_input_mfs(
    x: np.ndarray,
    feature_names: list[str],
    max_rules: int | None,
    sigma_scale: float,
    random_state: int | None,
) -> dict[str, list[GaussianMF]]:
    """Build point-based fuzzy rule base membership functions from training samples."""
    n_samples = x.shape[0]
    if max_rules is None or max_rules >= n_samples:
        indices = np.arange(n_samples)
    else:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_samples, size=int(max_rules), replace=False)
        indices = np.sort(indices)

    input_mfs: dict[str, list[GaussianMF]] = {}
    for d, name in enumerate(feature_names):
        col = x[:, d]
        sigma = max(float(np.std(col)) * sigma_scale, 1e-3)
        centers = col[indices]
        input_mfs[name] = [GaussianMF(mean=float(c), sigma=sigma) for c in centers]
    return input_mfs


def _wrap_dimension_dependent_gaussian_input_mfs(
    input_mfs: dict[str, list[GaussianMF]],
    dimension: int,
    xi: float = 745.0,
    rho: float | None = None,
) -> dict[str, list[GaussianMF]]:
    return {
        name: cast(
            list[GaussianMF],
            [
                DimensionDependentGaussianMF(
                    mean=mf.mean.detach().item(),
                    sigma=mf.sigma.detach().item(),
                    dimension=dimension,
                    xi=xi,
                    rho=rho,
                )
                for mf in mfs
            ],
        )
        for name, mfs in input_mfs.items()
    }


def _wrap_composite_gaussian_input_mfs(
    input_mfs: dict[str, list[GaussianMF]],
    eps: float | None = None,
) -> dict[str, list[GaussianMF]]:
    return {
        name: cast(
            list[GaussianMF],
            [
                CompositeGMF(
                    mean=mf.mean.detach().item(),
                    sigma=mf.sigma.detach().item(),
                    eps=eps if eps is not None else mf.eps,
                )
                for mf in mfs
            ],
        )
        for name, mfs in input_mfs.items()
    }


class _BaseClassifierEstimator(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Abstract base class for all highFIS TSK classifier estimators.

    Implements the full scikit-learn estimator protocol — ``fit``,
    ``predict_proba``, ``predict``, ``score``, ``save`` and ``load`` — and
    delegates only model construction to concrete subclasses via the abstract
    method :meth:`_build_model`.

    Subclasses must **not** override ``fit``; they should only implement
    ``_build_model`` (and optionally their own ``__init__`` to add extra
    hyperparameters).

    Attributes:
        model_: Fitted :class:`~highfis.base.BaseTSK` instance. Available
            after :meth:`fit`.
        classes_: Unique class labels discovered during :meth:`fit`.
        n_features_in_: Number of input features seen during :meth:`fit`.
        feature_names_in_: Array of feature name strings.
        history_: Training history dictionary returned by the underlying model.
        rule_base_: Rule-base type actually used during the last :meth:`fit`.
    """

    model_: BaseTSK

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise shared hyperparameters for TSK classifier estimators.

        Args:
            input_configs: Optional list of :class:`InputConfig` instances,
                one per input feature. Must match the number of columns in
                ``X`` when supplied. When ``mf_init="kmeans"`` only the
                ``name`` field is used; centres and sigmas are computed from
                cluster statistics.
            n_mfs: Number of MFs per feature when ``mf_init="grid"``, or
                number of k-means clusters when ``mf_init="kmeans"``. Cui
                et al. (IJCNN 2021) used ``R=30`` for all datasets.
            mf_init: MF initialisation strategy. ``"kmeans"`` (default)
                derives MF centres from k-means cluster centroids following
                Cui et al. (IJCNN 2021). ``"grid"`` places centres on a
                regular grid controlled by :class:`InputConfig`.
            sigma_scale: Scale factor for sigma initialisation when
                ``mf_init="kmeans"``. Each sigma is drawn from
                ``N(h, 0.2)`` where ``h = sigma_scale * within_cluster_std``
                (Cui et al., IJCNN 2021). Pass ``"auto"`` to set
                ``sigma_scale = sqrt(D)`` as recommended for vanilla TSK on
                high-dimensional data; HTSK and LogTSK handle dimensionality
                internally and use ``1.0``.
            random_state: Integer seed forwarded to k-means initialisation
                and ``torch.manual_seed``. Ensures reproducible runs.
            epochs: Maximum number of full passes over the training data.
                Training may stop earlier if ``patience`` is exhausted.
            learning_rate: Initial learning rate for the Adam optimiser.
                Cui et al. (IJCNN 2021) selected ``0.01`` via cross-
                validation across most datasets.
            verbose: Verbosity level. ``0`` = quiet, ``1`` = progress bar,
                ``2`` = per-epoch summary, ``3`` = full per-epoch logging.
                ``True`` is accepted as an alias for ``2``.
            rule_base: Explicit rule-base construction type. ``"coco"``
                (compactly combined) pairs rule ``r`` with MF ``r`` on every
                feature. ``"cartesian"`` enumerates all MF combinations.
                ``"pfrb"`` builds a point-based FRB from training samples and
                uses a CoCo rule base over the resulting sample-centered MFs.
                Defaults to ``"coco"`` for ``mf_init="kmeans"`` and
                ``"cartesian"`` for ``mf_init="grid"``.
            batch_size: Mini-batch size for gradient descent. Cui et al.
                (IJCNN 2021) used ``512`` (or ``min(N, 60)`` when the
                training set is smaller). ``None`` uses the full dataset.
            shuffle: If ``True``, training samples are reshuffled before
                each epoch.
            ur_weight: Weight of the uncertainty regularisation (UR) term
                added to the cross-entropy loss. ``0.0`` disables UR.
                Cui et al. (TFS 2020) describe the UR formulation.
            ur_target: Optional target firing-level for UR. ``None`` uses
                the model default.
            consequent_batch_norm: Apply batch normalisation to the
                consequent linear layers. Can improve training stability on
                large datasets.
            pfrb_max_rules: Maximum number of rules when using the point-
                based FRB (P-FRB) initialisation introduced by DG-TSK (Xue
                et al., Fuzzy Sets and Systems, 2023). ``None`` uses all
                training samples as rule prototypes.
            patience: Number of consecutive epochs without improvement on
                the validation loss before training is stopped early. Only
                active when ``validation_data`` is provided.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` tuple used for
                early stopping and held-out performance monitoring.
            weight_decay: L2 weight-decay coefficient applied to consequent
                layer parameters by the Adam optimiser.
        """
        self.input_configs = input_configs
        self.n_mfs = n_mfs
        self.mf_init = mf_init
        self.sigma_scale = sigma_scale
        self.random_state = random_state
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.rule_base = rule_base
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ur_weight = ur_weight
        self.ur_target = ur_target
        self.consequent_batch_norm = consequent_batch_norm
        self.pfrb_max_rules = pfrb_max_rules
        self.patience = patience
        self.restore_best = restore_best
        self.validation_data = validation_data
        self.weight_decay = weight_decay

    # -- helpers ----------------------------------------------------------

    def _resolve_input_configs(self, x: np.ndarray) -> list[InputConfig]:
        """Resolve per-feature input configs."""
        if self.input_configs is not None:
            if len(self.input_configs) != x.shape[1]:
                raise ValueError(
                    f"input_configs length ({len(self.input_configs)}) must match number of features ({x.shape[1]})"
                )
            return list(self.input_configs)
        return [InputConfig(name=f"x{i + 1}", n_mfs=int(self.n_mfs)) for i in range(x.shape[1])]

    def _resolve_feature_names(self, x: np.ndarray) -> list[str]:
        """Resolve feature names from configs or defaults."""
        if self.input_configs is not None:
            if len(self.input_configs) != x.shape[1]:
                raise ValueError(
                    f"input_configs length ({len(self.input_configs)}) must match number of features ({x.shape[1]})"
                )
            return [cfg.name for cfg in self.input_configs]
        return [f"x{i + 1}" for i in range(x.shape[1])]

    @staticmethod
    def _as_tensor_x(x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to float32 tensor."""
        return torch.as_tensor(x, dtype=torch.float32)

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[dict[str, list[GaussianMF]], list[str], str]:
        """Build MFs and resolve rule_base from the initialization mode."""
        init = str(self.mf_init).lower()
        if init not in {"kmeans", "grid"}:
            raise ValueError(f"mf_init must be 'kmeans' or 'grid', got '{self.mf_init}'")

        if init == "kmeans":
            feature_names = self._resolve_feature_names(x_arr)
            if isinstance(self.sigma_scale, str) and self.sigma_scale.lower() == "auto":
                effective_sigma_scale = math.sqrt(float(x_arr.shape[1]))
            else:
                effective_sigma_scale = float(self.sigma_scale)
            if self.rule_base == "pfrb":
                input_mfs = _build_pfrb_input_mfs(
                    x_arr,
                    feature_names,
                    max_rules=self.pfrb_max_rules,
                    sigma_scale=effective_sigma_scale,
                    random_state=self.random_state,
                )
                effective_rule_base = "coco"
            else:
                input_mfs = _build_kmeans_input_mfs(
                    x_arr,
                    n_clusters=int(self.n_mfs),
                    sigma_scale=effective_sigma_scale,
                    feature_names=feature_names,
                    random_state=self.random_state,
                )
                effective_rule_base = self.rule_base if self.rule_base is not None else "coco"
        else:
            input_configs = self._resolve_input_configs(x_arr)
            feature_names = [cfg.name for cfg in input_configs]
            if self.rule_base == "pfrb":
                input_mfs = _build_pfrb_input_mfs(
                    x_arr,
                    feature_names,
                    max_rules=self.pfrb_max_rules,
                    sigma_scale=float(self.sigma_scale) if not isinstance(self.sigma_scale, str) else 1.0,
                    random_state=self.random_state,
                )
                effective_rule_base = "coco"
            else:
                input_mfs = _build_gaussian_input_mfs(x_arr, input_configs)
                effective_rule_base = self.rule_base if self.rule_base is not None else "cartesian"

        return input_mfs, feature_names, effective_rule_base

    @abstractmethod
    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create the concrete TSK classification model."""

    # -- sklearn API ------------------------------------------------------

    def fit(self, x: Any, y: Any) -> Self:
        """Train the TSK classifier on labeled samples."""
        x_arr, y_arr = check_X_y(x, y)

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        le = LabelEncoder()
        y_idx = le.fit_transform(np.asarray(y_arr))

        input_mfs, feature_names, effective_rule_base = self._build_input_mfs(x_arr)

        self.n_features_in_ = x_arr.shape[1]
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.classes_ = le.classes_
        self._label_encoder_ = le

        self.model_ = self._build_model(input_mfs, len(self.classes_), effective_rule_base)

        y_t = torch.as_tensor(y_idx, dtype=torch.long)

        # Prepare validation tensors if provided
        x_val_t: torch.Tensor | None = None
        y_val_t: torch.Tensor | None = None
        if self.validation_data is not None:
            x_v, y_v = self.validation_data
            x_v_arr = check_array(x_v)
            y_v_idx = self._label_encoder_.transform(np.asarray(y_v))
            x_val_t = self._as_tensor_x(x_v_arr)
            y_val_t = torch.as_tensor(y_v_idx, dtype=torch.long)

        self.history_ = self.model_.fit(
            self._as_tensor_x(x_arr),
            y_t,
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            batch_size=self.batch_size,
            shuffle=bool(self.shuffle),
            ur_weight=float(self.ur_weight),
            ur_target=self.ur_target,
            verbose=self.verbose,
            x_val=x_val_t,
            y_val=y_val_t,
            patience=self.patience,
            restore_best=self.restore_best,
            weight_decay=float(self.weight_decay),
        )
        self.rule_base_ = effective_rule_base
        return self

    def _build_checkpoint_base(
        self,
        *,
        model_init: dict[str, Any],
        fitted_attrs: dict[str, Any],
    ) -> dict[str, Any]:
        check_is_fitted(self, "model_")
        return {
            "format": CHECKPOINT_FORMAT,
            "format_version": CHECKPOINT_VERSION,
            "estimator_class": self.__class__.__name__,
            "estimator_params": self.get_params(deep=False),
            "model_init": model_init,
            "model_state_dict": self.model_.state_dict(),
            "fitted_attrs": fitted_attrs,
            "history": getattr(self, "history_", None),
        }

    def save(self, path: str) -> None:
        """Persist estimator configuration, model weights and fitted metadata."""
        checkpoint = self._build_checkpoint_base(
            model_init={
                "input_mfs": self.model_.input_mfs,
                "n_classes": len(self.classes_),
                "rule_base": self.rule_base_,
            },
            fitted_attrs={
                "n_features_in": int(self.n_features_in_),
                "feature_names_in": self.feature_names_in_.tolist(),
                "classes": self.classes_.tolist(),
            },
        )
        save_checkpoint(path, checkpoint)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load a persisted estimator created by save."""
        checkpoint = load_checkpoint(path)
        validate_checkpoint_payload(checkpoint, expected_estimator_class=cls.__name__)

        estimator = cls(**checkpoint["estimator_params"])
        model_init = checkpoint["model_init"]
        estimator.rule_base_ = model_init["rule_base"]
        estimator.model_ = estimator._build_model(
            model_init["input_mfs"],
            int(model_init["n_classes"]),
            str(model_init["rule_base"]),
        )
        estimator.model_.load_state_dict(checkpoint["model_state_dict"])

        fitted = checkpoint["fitted_attrs"]
        estimator.n_features_in_ = int(fitted["n_features_in"])
        estimator.feature_names_in_ = np.asarray(fitted["feature_names_in"], dtype=object)
        estimator.classes_ = np.asarray(fitted["classes"], dtype=object)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = estimator.classes_
        estimator._label_encoder_ = label_encoder
        estimator.history_ = cast(dict[str, Any], checkpoint.get("history", {}))
        return estimator

    def predict_proba(self, x: Any) -> np.ndarray:
        """Predict class probabilities for input samples."""
        check_is_fitted(self, "model_")
        x_arr = check_array(x)
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_arr.shape[1]}")
        probs = cast(Any, self.model_).predict_proba(self._as_tensor_x(x_arr))
        return probs.detach().cpu().numpy()

    def predict(self, x: Any) -> np.ndarray:
        """Predict class labels for input samples."""
        proba = self.predict_proba(x)
        y_idx = np.argmax(proba, axis=1)
        return np.asarray(self._label_encoder_.inverse_transform(y_idx))

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return classification accuracy on the provided dataset."""
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))

    def evaluate(
        self,
        X: Any,
        y: Any,
        metrics: list[str] | None = None,
        sample_weight: Any | None = None,
    ) -> dict[str, float]:
        """Compute classification evaluation metrics for the provided dataset."""
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return compute_metrics(
            task="classification",
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            metrics=metrics,
        )


class _BaseRegressorEstimator(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Abstract base class for all highFIS TSK regressor estimators.

    Implements the full scikit-learn estimator protocol — ``fit``,
    ``predict``, ``score``, ``save`` and ``load`` — and delegates only model
    construction to concrete subclasses via the abstract method
    :meth:`_build_model`.

    Subclasses must **not** override ``fit``; they should only implement
    ``_build_model`` (and optionally their own ``__init__`` to add extra
    hyperparameters).

    Attributes:
        model_: Fitted :class:`~highfis.base.BaseTSK` instance. Available
            after :meth:`fit`.
        n_features_in_: Number of input features seen during :meth:`fit`.
        feature_names_in_: Array of feature name strings.
        history_: Training history dictionary returned by the underlying model.
        rule_base_: Rule-base type actually used during the last :meth:`fit`.
    """

    model_: BaseTSK

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise shared hyperparameters for TSK regressor estimators.

        Args:
            input_configs: Optional list of :class:`InputConfig` instances,
                one per input feature. Must match the number of columns in
                ``X`` when supplied. When ``mf_init="kmeans"`` only the
                ``name`` field is used; centres and sigmas are computed from
                cluster statistics.
            n_mfs: Number of MFs per feature when ``mf_init="grid"``, or
                number of k-means clusters when ``mf_init="kmeans"``. Cui
                et al. (IJCNN 2021) used ``R=30`` for all datasets.
            mf_init: MF initialisation strategy. ``"kmeans"`` (default)
                derives MF centres from k-means cluster centroids following
                Cui et al. (IJCNN 2021). ``"grid"`` places centres on a
                regular grid controlled by :class:`InputConfig`.
            sigma_scale: Scale factor for sigma initialisation when
                ``mf_init="kmeans"``. Each sigma is drawn from
                ``N(h, 0.2)`` where ``h = sigma_scale * within_cluster_std``
                (Cui et al., IJCNN 2021). Pass ``"auto"`` to set
                ``sigma_scale = sqrt(D)`` as recommended for vanilla TSK on
                high-dimensional data; HTSK and LogTSK handle dimensionality
                internally and use ``1.0``.
            random_state: Integer seed forwarded to k-means initialisation
                and ``torch.manual_seed``. Ensures reproducible runs.
            epochs: Maximum number of full passes over the training data.
                Training may stop earlier if ``patience`` is exhausted.
            learning_rate: Initial learning rate for the Adam optimiser.
                Cui et al. (IJCNN 2021) selected ``0.01`` via cross-
                validation across most datasets.
            verbose: Verbosity level. ``0`` = quiet, ``1`` = progress bar,
                ``2`` = per-epoch summary, ``3`` = full per-epoch logging.
                ``True`` is accepted as an alias for ``2``.
            rule_base: Explicit rule-base construction type. ``"coco"``
                (compactly combined) pairs rule ``r`` with MF ``r`` on every
                feature. ``"cartesian"`` enumerates all MF combinations.
                ``"pfrb"`` builds a point-based FRB from training samples and
                uses a CoCo rule base over the resulting sample-centered MFs.
                Defaults to ``"coco"`` for ``mf_init="kmeans"`` and
                ``"cartesian"`` for ``mf_init="grid"``.
            batch_size: Mini-batch size for gradient descent. Cui et al.
                (IJCNN 2021) used ``512`` (or ``min(N, 60)`` when the
                training set is smaller). ``None`` uses the full dataset.
            shuffle: If ``True``, training samples are reshuffled before
                each epoch.
            ur_weight: Weight of the uncertainty regularisation (UR) term
                added to the MSE loss. ``0.0`` disables UR.
            ur_target: Optional target firing-level for UR. ``None`` uses
                the model default.
            consequent_batch_norm: Apply batch normalisation to the
                consequent linear layers. Can improve training stability on
                large datasets.
            pfrb_max_rules: Maximum number of point-based FRB rules when
                ``rule_base='pfrb'``. ``None`` uses all training samples.
            patience: Number of consecutive epochs without improvement on
                the validation loss before training is stopped early. Only
                active when ``validation_data`` is provided.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` tuple used for
                early stopping and held-out performance monitoring.
            weight_decay: L2 weight-decay coefficient applied to consequent
                layer parameters by the Adam optimiser.
        """
        self.input_configs = input_configs
        self.n_mfs = n_mfs
        self.mf_init = mf_init
        self.sigma_scale = sigma_scale
        self.random_state = random_state
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.rule_base = rule_base
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ur_weight = ur_weight
        self.ur_target = ur_target
        self.consequent_batch_norm = consequent_batch_norm
        self.pfrb_max_rules = pfrb_max_rules
        self.patience = patience
        self.restore_best = restore_best
        self.validation_data = validation_data
        self.weight_decay = weight_decay

    # -- helpers ----------------------------------------------------------

    def _resolve_input_configs(self, x: np.ndarray) -> list[InputConfig]:
        """Resolve per-feature input configs."""
        if self.input_configs is not None:
            if len(self.input_configs) != x.shape[1]:
                raise ValueError(
                    f"input_configs length ({len(self.input_configs)}) must match number of features ({x.shape[1]})"
                )
            return list(self.input_configs)
        return [InputConfig(name=f"x{i + 1}", n_mfs=int(self.n_mfs)) for i in range(x.shape[1])]

    def _resolve_feature_names(self, x: np.ndarray) -> list[str]:
        """Resolve feature names from configs or defaults."""
        if self.input_configs is not None:
            if len(self.input_configs) != x.shape[1]:
                raise ValueError(
                    f"input_configs length ({len(self.input_configs)}) must match number of features ({x.shape[1]})"
                )
            return [cfg.name for cfg in self.input_configs]
        return [f"x{i + 1}" for i in range(x.shape[1])]

    @staticmethod
    def _as_tensor_x(x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to float32 tensor."""
        return torch.as_tensor(x, dtype=torch.float32)

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[dict[str, list[GaussianMF]], list[str], str]:
        """Build MFs and resolve rule_base from the initialization mode."""
        init = str(self.mf_init).lower()
        if init not in {"kmeans", "grid"}:
            raise ValueError(f"mf_init must be 'kmeans' or 'grid', got '{self.mf_init}'")

        if init == "kmeans":
            feature_names = self._resolve_feature_names(x_arr)
            if isinstance(self.sigma_scale, str) and self.sigma_scale.lower() == "auto":
                effective_sigma_scale = math.sqrt(float(x_arr.shape[1]))
            else:
                effective_sigma_scale = float(self.sigma_scale)
            if self.rule_base == "pfrb":
                input_mfs = _build_pfrb_input_mfs(
                    x_arr,
                    feature_names,
                    max_rules=self.pfrb_max_rules,
                    sigma_scale=effective_sigma_scale,
                    random_state=self.random_state,
                )
                effective_rule_base = "coco"
            else:
                input_mfs = _build_kmeans_input_mfs(
                    x_arr,
                    n_clusters=int(self.n_mfs),
                    sigma_scale=effective_sigma_scale,
                    feature_names=feature_names,
                    random_state=self.random_state,
                )
                effective_rule_base = self.rule_base if self.rule_base is not None else "coco"
        else:
            input_configs = self._resolve_input_configs(x_arr)
            feature_names = [cfg.name for cfg in input_configs]
            if self.rule_base == "pfrb":
                input_mfs = _build_pfrb_input_mfs(
                    x_arr,
                    feature_names,
                    max_rules=self.pfrb_max_rules,
                    sigma_scale=float(self.sigma_scale) if not isinstance(self.sigma_scale, str) else 1.0,
                    random_state=self.random_state,
                )
                effective_rule_base = "coco"
            else:
                input_mfs = _build_gaussian_input_mfs(x_arr, input_configs)
                effective_rule_base = self.rule_base if self.rule_base is not None else "cartesian"

        return input_mfs, feature_names, effective_rule_base

    @abstractmethod
    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create the concrete TSK regression model."""

    # -- sklearn API ------------------------------------------------------

    def fit(self, x: Any, y: Any) -> Self:
        """Train the TSK regressor on labeled samples."""
        x_arr, y_arr = check_X_y(x, y)

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        input_mfs, feature_names, effective_rule_base = self._build_input_mfs(x_arr)

        self.n_features_in_ = x_arr.shape[1]
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)

        self.model_ = self._build_regressor_model(input_mfs, effective_rule_base)

        y_t = torch.as_tensor(np.asarray(y_arr, dtype=np.float32), dtype=torch.float32)

        # Prepare validation tensors if provided
        x_val_t: torch.Tensor | None = None
        y_val_t: torch.Tensor | None = None
        if self.validation_data is not None:
            x_v, y_v = self.validation_data
            x_v_arr = check_array(x_v)
            x_val_t = self._as_tensor_x(x_v_arr)
            y_val_t = torch.as_tensor(np.asarray(y_v, dtype=np.float32), dtype=torch.float32)

        self.history_ = self.model_.fit(
            self._as_tensor_x(x_arr),
            y_t,
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            batch_size=self.batch_size,
            shuffle=bool(self.shuffle),
            ur_weight=float(self.ur_weight),
            ur_target=self.ur_target,
            verbose=self.verbose,
            x_val=x_val_t,
            y_val=y_val_t,
            patience=self.patience,
            restore_best=self.restore_best,
            weight_decay=float(self.weight_decay),
        )
        self.rule_base_ = effective_rule_base
        return self

    def _build_checkpoint_base(
        self,
        *,
        model_init: dict[str, Any],
        fitted_attrs: dict[str, Any],
    ) -> dict[str, Any]:
        check_is_fitted(self, "model_")
        return {
            "format": CHECKPOINT_FORMAT,
            "format_version": CHECKPOINT_VERSION,
            "estimator_class": self.__class__.__name__,
            "estimator_params": self.get_params(deep=False),
            "model_init": model_init,
            "model_state_dict": self.model_.state_dict(),
            "fitted_attrs": fitted_attrs,
            "history": getattr(self, "history_", None),
        }

    def save(self, path: str) -> None:
        """Persist estimator configuration, model weights and fitted metadata."""
        checkpoint = self._build_checkpoint_base(
            model_init={
                "input_mfs": self.model_.input_mfs,
                "rule_base": self.rule_base_,
            },
            fitted_attrs={
                "n_features_in": int(self.n_features_in_),
                "feature_names_in": self.feature_names_in_.tolist(),
            },
        )
        save_checkpoint(path, checkpoint)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load a persisted estimator created by save."""
        checkpoint = load_checkpoint(path)
        validate_checkpoint_payload(checkpoint, expected_estimator_class=cls.__name__)

        estimator = cls(**checkpoint["estimator_params"])
        model_init = checkpoint["model_init"]
        estimator.rule_base_ = model_init["rule_base"]
        estimator.model_ = estimator._build_regressor_model(
            model_init["input_mfs"],
            str(model_init["rule_base"]),
        )
        estimator.model_.load_state_dict(checkpoint["model_state_dict"])

        fitted = checkpoint["fitted_attrs"]
        estimator.n_features_in_ = int(fitted["n_features_in"])
        estimator.feature_names_in_ = np.asarray(fitted["feature_names_in"], dtype=object)
        estimator.history_ = cast(dict[str, Any], checkpoint.get("history", {}))
        return estimator

    def predict(self, x: Any) -> np.ndarray:
        """Predict continuous target values for input samples."""
        check_is_fitted(self, "model_")
        x_arr = check_array(x)
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_arr.shape[1]}")
        preds = cast(Any, self.model_).predict(self._as_tensor_x(x_arr))
        return preds.detach().cpu().numpy()

    def evaluate(
        self,
        X: Any,
        y: Any,
        metrics: list[str] | None = None,
        sample_weight: Any | None = None,
    ) -> dict[str, float]:
        """Compute regression evaluation metrics for the provided dataset."""
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return compute_metrics(
            task="regression",
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            metrics=metrics,
        )


# =====================================================================
# HTSK Estimators
# =====================================================================


class HTSKClassifierEstimator(_BaseClassifierEstimator):
    r"""HTSK classifier for high-dimensional TSK inference.

    HTSK replaces the standard product t-norm with a geometric mean over
    membership values and performs rule normalization in log-space.

    References:
        Y. Cui, D. Wu and Y. Xu, "Curse of Dimensionality for TSK Fuzzy
        Neural Networks: Explanation and Solutions," 2021 International
        Joint Conference on Neural Networks (IJCNN), Shenzhen, China,
        2021, pp. 1-8, doi: 10.1109/IJCNN52387.2021.9534265.

    Example:
        ```python
        from highfis import HTSKClassifierEstimator

        clf = HTSKClassifierEstimator()
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 3,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an HTSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs.
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` is recommended for HTSK.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by HTSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create HTSKClassifier."""
        return HTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class HTSKRegressorEstimator(_BaseRegressorEstimator):
    r"""HTSK regressor for high-dimensional TSK inference.

    HTSK replaces the standard product t-norm with a geometric mean over
    membership values and performs rule normalization in log-space.

    References:
        Y. Cui, D. Wu and Y. Xu, "Curse of Dimensionality for TSK Fuzzy
        Neural Networks: Explanation and Solutions," 2021 International
        Joint Conference on Neural Networks (IJCNN), Shenzhen, China,
        2021, pp. 1-8, doi: 10.1109/IJCNN52387.2021.9534265.

    Example:
        ```python
        from highfis import HTSKRegressorEstimator

        reg = HTSKRegressorEstimator()
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 3,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an HTSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs.
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Scale factor for sigma initialisation when
                ``mf_init="kmeans"``. ``1.0`` is recommended for HTSK.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by HTSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create HTSKRegressor."""
        return HTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


# =====================================================================
# Vanilla TSK Estimators  (Takagi & Sugeno, 1985)
# =====================================================================


class TSKClassifierEstimator(_BaseClassifierEstimator):
    r"""Vanilla TSK classifier with sum-based rule normalization.

    The vanilla Takagi-Sugeno-Kang inference computes rule firing strengths
    with the product t-norm and normalizes them by their total sum.

    References:
        T. Takagi and M. Sugeno, "Fuzzy identification of systems and
        its applications to modeling and control," in IEEE
        Transactions on Systems, Man, and Cybernetics, vol. SMC-15,
        no. 1, pp. 116-132, Jan.-Feb. 1985,
        doi: 10.1109/TSMC.1985.6313399.

    Example:
        ```python
        from highfis import TSKClassifierEstimator

        clf = TSKClassifierEstimator(n_mfs=5, random_state=0)
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a vanilla TSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. Use ``"auto"`` (= ``sqrt(D)``)
                for high-dimensional data to mitigate softmax saturation
                (Cui et al., IJCNN 2021). ``1.0`` is appropriate for low-
                to medium-dimensional problems.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by TSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create TSKClassifier."""
        return TSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class TSKRegressorEstimator(_BaseRegressorEstimator):
    r"""Vanilla TSK regressor with sum-based rule normalization.

    The vanilla Takagi-Sugeno-Kang inference computes rule firing strengths
    with the product t-norm and normalizes them by their total sum.

    References:
        T. Takagi and M. Sugeno, "Fuzzy identification of systems and
        its applications to modeling and control," in IEEE
        Transactions on Systems, Man, and Cybernetics, vol. SMC-15,
        no. 1, pp. 116-132, Jan.-Feb. 1985,
        doi: 10.1109/TSMC.1985.6313399.

    Example:
        ```python
        from highfis import TSKRegressorEstimator

        reg = TSKRegressorEstimator(n_mfs=30, random_state=0)
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a vanilla TSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. Use ``"auto"`` (= ``sqrt(D)``)
                to mitigate softmax saturation on high-dimensional data.
                ``1.0`` is appropriate for low-to-medium-dimensional problems.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        return TSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class HDFISProdClassifierEstimator(_BaseClassifierEstimator):
    r"""HDFIS-prod classifier estimator with dimension-dependent Gaussian MFs.

    HDFIS-prod combines the standard product T-norm with a dimension-dependent
    Gaussian membership function (DMF) to avoid numeric underflow in very
    high-dimensional feature spaces while preserving first-order TSK
    consequents.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.

    Example:
        ```python
        from highfis import HDFISProdClassifierEstimator

        clf = HDFISProdClassifierEstimator()
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
        xi: float = 745.0,
        rho: float | None = None,
    ) -> None:
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )
        self.xi = float(xi)
        self.rho = rho

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[dict[str, list[GaussianMF]], list[str], str]:
        input_mfs, feature_names, effective_rule_base = super()._build_input_mfs(x_arr)
        return (
            _wrap_dimension_dependent_gaussian_input_mfs(
                input_mfs,
                dimension=x_arr.shape[1],
                xi=self.xi,
                rho=self.rho,
            ),
            feature_names,
            effective_rule_base,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        return HDFISProdClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class HDFISProdRegressorEstimator(_BaseRegressorEstimator):
    r"""HDFIS-prod regressor estimator with dimension-dependent Gaussian MFs.

    HDFIS-prod combines the standard product T-norm with a dimension-dependent
    Gaussian membership function (DMF) to avoid numeric underflow in very
    high-dimensional feature spaces while preserving first-order TSK
    consequents.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.

    Example:
        ```python
        from highfis import HDFISProdRegressorEstimator

        reg = HDFISProdRegressorEstimator()
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
        xi: float = 745.0,
        rho: float | None = None,
    ) -> None:
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )
        self.xi = float(xi)
        self.rho = rho

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[dict[str, list[GaussianMF]], list[str], str]:
        input_mfs, feature_names, effective_rule_base = super()._build_input_mfs(x_arr)
        return (
            _wrap_dimension_dependent_gaussian_input_mfs(
                input_mfs,
                dimension=x_arr.shape[1],
                xi=self.xi,
                rho=self.rho,
            ),
            feature_names,
            effective_rule_base,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        return HDFISProdRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class HDFISMinClassifierEstimator(_BaseClassifierEstimator):
    r"""HDFIS-min classifier estimator with minimum T-norm antecedents.

    HDFIS-min freezes antecedent membership parameters and uses a minimum
    T-norm aggregation in the antecedent, so that only consequent parameters
    are optimized during training. This matches the paper's observation that
    minimum-based high-dimensional inference is best handled by fixing the
    antecedent structure and training the rule consequents.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.

    Example:
        ```python
        from highfis import HDFISMinClassifierEstimator

        clf = HDFISMinClassifierEstimator()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an HDFIS-min classifier estimator."""
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        return HDFISMinClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class HDFISMinRegressorEstimator(_BaseRegressorEstimator):
    r"""HDFIS-min regressor estimator with minimum T-norm antecedents.

    HDFIS-min freezes antecedent membership parameters and uses a minimum
    T-norm aggregation in the antecedent, so that only consequent parameters
    are optimized during training. This design avoids the nondifferentiability
    of the minimum operator while preserving first-order TSK consequents.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.

    Example:
        ```python
        from highfis import HDFISMinRegressorEstimator

        reg = HDFISMinRegressorEstimator()
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an HDFIS-min regressor estimator."""
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        return HDFISMinRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class AYATSKClassifierEstimator(_BaseClassifierEstimator):
    """TSK classifier with an adaptive Yager T-norm in the antecedent.

    AYATSK extends TSK by using an adaptive Yager T-norm aggregation and
    optional positive lower-bound membership functions to improve
    stability and performance in high-dimensional settings.

    Reference:
        G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-Norm-Based
        Takagi-Sugeno-Kang Fuzzy Systems," in IEEE Transactions on
        Systems, Man, and Cybernetics: Systems, vol. 55, no. 12,
        pp. 9802-9815, Dec. 2025, doi: 10.1109/TSMC.2025.3621346.

    Example:
        ```python
        from highfis import AYATSKClassifierEstimator

        clf = AYATSKClassifierEstimator(n_mfs=30, random_state=0)
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an AYATSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor for k-means initialisation.
                ``1.0`` is recommended; the adaptive Yager T-norm handles
                high-dimensional stability internally.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by
                AYATSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create AYATSKClassifier."""
        return AYATSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class AYATSKRegressorEstimator(_BaseRegressorEstimator):
    """TSK regressor with an adaptive Yager T-norm in the antecedent.

    AYATSK extends TSK by using an adaptive Yager T-norm aggregation and
    optional positive lower-bound membership functions to improve
    stability and performance in high-dimensional settings.

    Reference:
        G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-Norm-Based
        Takagi-Sugeno-Kang Fuzzy Systems," in IEEE Transactions on
        Systems, Man, and Cybernetics: Systems, vol. 55, no. 12,
        pp. 9802-9815, Dec. 2025, doi: 10.1109/TSMC.2025.3621346.

    Example:
        ```python
        from highfis import AYATSKRegressorEstimator

        reg = AYATSKRegressorEstimator(n_mfs=30, random_state=0)
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an AYATSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create AYATSKRegressor."""
        return AYATSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class DombiTSKClassifierEstimator(_BaseClassifierEstimator):
    r"""TSK classifier with a fixed Dombi T-norm in the antecedent.

    DombiTSK extends TSK fuzzy inference by using a Dombi t-norm
    aggregation in antecedent evaluation while keeping first-order
    linear consequents.

    Reference:
        G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: A
        High-Dimensional Takagi-Sugeno-Kang Fuzzy System Based on
        Adaptive Dombi T-Norm," in IEEE Transactions on Fuzzy
        Systems, vol. 33, no. 6, pp. 1767-1780, June 2025,
        doi: 10.1109/TFUZZ.2025.3535640.

    Example:
        ```python
        from highfis import DombiTSKClassifierEstimator

        clf = DombiTSKClassifierEstimator(n_mfs=30, random_state=0)
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a DombiTSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended; the Dombi
                T-norm handles high-dimensional stability without inflating
                sigma.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by
                DombiTSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create DombiTSKClassifier."""
        return DombiTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class DombiTSKRegressorEstimator(_BaseRegressorEstimator):
    r"""TSK regressor with a fixed Dombi T-norm in the antecedent.

    DombiTSK extends TSK fuzzy inference by using a Dombi t-norm
    aggregation in antecedent evaluation while keeping first-order
    linear consequents.

    Reference:
        G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: A
        High-Dimensional Takagi-Sugeno-Kang Fuzzy System Based on
        Adaptive Dombi T-Norm," in IEEE Transactions on Fuzzy
        Systems, vol. 33, no. 6, pp. 1767-1780, June 2025,
        doi: 10.1109/TFUZZ.2025.3535640.

    Example:
        ```python
        from highfis import DombiTSKRegressorEstimator

        reg = DombiTSKRegressorEstimator(n_mfs=30, random_state=0)
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a DombiTSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create DombiTSKRegressor."""
        return DombiTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class ADMTSKClassifierEstimator(_BaseClassifierEstimator):
    r"""ADMTSK classifier estimator with Composite GMF and adaptive Dombi lambda.

    ADMTSK is an adaptive Dombi TSK fuzzy system designed for high-dimensional inference.
    It combines a Dombi T-norm antecedent with a positive lower-bound Composite Gaussian
    membership function (CGMF) and normalized first-order consequents.

    Reference:
        G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: A High-Dimensional
        Takagi-Sugeno-Kang Fuzzy System Based on Adaptive Dombi T-Norm," in IEEE
        Transactions on Fuzzy Systems, vol. 33, no. 6, pp. 1767-1780, June 2025,
        doi: 10.1109/TFUZZ.2025.3535640.

    Example:
        ```python
        from highfis import ADMTSKClassifierEstimator

        clf = ADMTSKClassifierEstimator()
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
        adaptive: bool = True,
        lambda_: float = 1.0,
        lower_bound: float = 1.0 / math.e,
        K: float = 10.0,
    ) -> None:
        """Initialize an ADMTSK classifier estimator.

        Args:
            input_configs: Optional list of per-feature input configurations.
            n_mfs: Number of membership functions per input when using
                ``mf_init="kmeans"`` or ``mf_init="grid"``.
            mf_init: Initialisation strategy for MFs, either ``"kmeans"``
                or ``"grid"``.
            sigma_scale: Scale factor used to initialise Gaussian MF sigma
                values.
            random_state: Random seed for MF initialisation and weights.
            epochs: Maximum number of training epochs.
            learning_rate: Learning rate for the optimizer.
            verbose: Verbosity level for training output.
            rule_base: Rule base strategy override, typically ``"coco"`` or
                ``"cartesian"``.
            batch_size: Mini-batch size for training.
            shuffle: Whether to shuffle training data each epoch.
            ur_weight: Uniform-rule regularisation weight.
            ur_target: Target average rule activation for uniform regularisation.
            consequent_batch_norm: If True, apply batch normalization to
                consequent inputs.
            pfrb_max_rules: Maximum number of rules for point-based FRB.
            patience: Early stopping patience. Use ``None`` to disable.
            restore_best: If True, restore the best validation weights.
            validation_data: Validation dataset used for early stopping.
            weight_decay: Weight decay applied during training.
            adaptive: If True, use adaptive lambda selection for Dombi T-norm.
            lambda_: Fixed Dombi parameter when adaptive is False.
            lower_bound: Lower bound used by Composite GMF.
            K: Heuristic constant used to compute adaptive lambda.

        Raises:
            ValueError: If estimator hyperparameters are invalid.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )
        self.adaptive = bool(adaptive)
        self.lambda_ = float(lambda_)
        self.lower_bound = float(lower_bound)
        self.K = float(K)

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[dict[str, list[GaussianMF]], list[str], str]:
        input_mfs, feature_names, effective_rule_base = super()._build_input_mfs(x_arr)
        return (
            _wrap_composite_gaussian_input_mfs(input_mfs),
            feature_names,
            effective_rule_base,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        return ADMTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            adaptive=self.adaptive,
            lambda_=self.lambda_,
            lower_bound=self.lower_bound,
            K=self.K,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class ADMTSKRegressorEstimator(_BaseRegressorEstimator):
    r"""ADMTSK regressor estimator with Composite GMF and adaptive Dombi lambda.

    ADMTSK is an adaptive Dombi TSK fuzzy system designed for high-dimensional inference.
    It combines a Dombi T-norm antecedent with a positive lower-bound Composite Gaussian
    membership function (CGMF) and normalized first-order consequents.

    Reference:
        G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: A High-Dimensional
        Takagi-Sugeno-Kang Fuzzy System Based on Adaptive Dombi T-Norm," in IEEE
        Transactions on Fuzzy Systems, vol. 33, no. 6, pp. 1767-1780, June 2025,
        doi: 10.1109/TFUZZ.2025.3535640.

    Example:
        ```python
        from highfis import ADMTSKRegressorEstimator

        reg = ADMTSKRegressorEstimator()
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
        adaptive: bool = True,
        lambda_: float = 1.0,
        lower_bound: float = 1.0 / math.e,
        K: float = 10.0,
    ) -> None:
        """Initialize an ADMTSK regressor estimator.

        Args:
            input_configs: Optional list of per-feature input configurations.
            n_mfs: Number of membership functions per input when using
                ``mf_init="kmeans"`` or ``mf_init="grid"``.
            mf_init: Initialisation strategy for MFs, either ``"kmeans"``
                or ``"grid"``.
            sigma_scale: Scale factor used to initialise Gaussian MF sigma
                values.
            random_state: Random seed for MF initialisation and weights.
            epochs: Maximum number of training epochs.
            learning_rate: Learning rate for the optimizer.
            verbose: Verbosity level for training output.
            rule_base: Rule base strategy override, typically ``"coco"`` or
                ``"cartesian"``.
            batch_size: Mini-batch size for training.
            shuffle: Whether to shuffle training data each epoch.
            ur_weight: Uniform-rule regularisation weight.
            ur_target: Target average rule activation for uniform regularisation.
            consequent_batch_norm: If True, apply batch normalization to
                consequent inputs.
            patience: Early stopping patience. Use ``None`` to disable.
            restore_best: If True, restore the best validation weights.
            validation_data: Validation dataset used for early stopping.
            weight_decay: Weight decay applied during training.
            adaptive: If True, use adaptive lambda selection for Dombi T-norm.
            lambda_: Fixed Dombi parameter when adaptive is False.
            lower_bound: Lower bound used by Composite GMF.
            K: Heuristic constant used to compute adaptive lambda.

        Raises:
            ValueError: If estimator hyperparameters are invalid.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )
        self.adaptive = bool(adaptive)
        self.lambda_ = float(lambda_)
        self.lower_bound = float(lower_bound)
        self.K = float(K)

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[dict[str, list[GaussianMF]], list[str], str]:
        input_mfs, feature_names, effective_rule_base = super()._build_input_mfs(x_arr)
        return (
            _wrap_composite_gaussian_input_mfs(input_mfs),
            feature_names,
            effective_rule_base,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        return ADMTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            adaptive=self.adaptive,
            lambda_=self.lambda_,
            lower_bound=self.lower_bound,
            K=self.K,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class AdaTSKClassifierEstimator(_BaseClassifierEstimator):
    r"""TSK classifier with adaptive softmin antecedent (AdaTSK).

    The firing strength of each rule is computed with the Ada-softmin operator.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import AdaTSKClassifierEstimator

        clf = AdaTSKClassifierEstimator(n_mfs=30, random_state=0)
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an AdaTSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended; Ada-softmin
                handles high-dimensional stability.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create AdaTSKClassifier."""
        return AdaTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class AdaTSKRegressorEstimator(_BaseRegressorEstimator):
    r"""TSK regressor with adaptive softmin antecedent (AdaTSK).

    The firing strength of each rule is computed with the Ada-softmin operator.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import AdaTSKRegressorEstimator

        reg = AdaTSKRegressorEstimator(n_mfs=30, random_state=0)
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an AdaTSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create AdaTSKRegressor."""
        return AdaTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class FSREAdaTSKClassifierEstimator(_BaseClassifierEstimator):
    r"""FSRE-AdaTSK classifier with adaptive softmin antecedent and gated consequents.

    FSRE-AdaTSK (Feature Selection and Rule Extraction) extends AdaTSK.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import FSREAdaTSKClassifierEstimator

        clf = FSREAdaTSKClassifierEstimator()
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        lambda_init: float = 1.0,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an FSRE-AdaTSK classifier.

        Args:
            lambda_init: Initial ALE-softmin parameter ``λ > 0`` inherited
                by :class:`DGALETSKClassifierEstimator`; not used by
                FSRE-AdaTSK proper (Ada-softmin computes its index from
                the current membership values). Default ``1.0``.
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) whose
                size grows linearly with the number of features, allowing
                more candidate rules for the RE phase. Xue et al. (2023)
                activate En-FRB after the FS phase; set ``False`` (default)
                to keep the compact CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        self.lambda_init = float(lambda_init)
        self.use_en_frb = bool(use_en_frb)
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create FSREAdaTSKClassifier."""
        return FSREAdaTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class FSREAdaTSKRegressorEstimator(_BaseRegressorEstimator):
    r"""FSRE-AdaTSK regressor with adaptive softmin antecedent and gated consequents.

    FSRE-AdaTSK (Feature Selection and Rule Extraction) extends AdaTSK.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import FSREAdaTSKRegressorEstimator

        reg = FSREAdaTSKRegressorEstimator()
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        lambda_init: float = 1.0,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an FSRE-AdaTSK regressor.

        Args:
            lambda_init: Initial ALE-softmin parameter ``λ > 0`` inherited
                by :class:`DGALETSKRegressorEstimator`; not used by
                FSRE-AdaTSK proper (Ada-softmin computes its index from
                the current membership values). Default ``1.0``.
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) for rule
                extraction. Default ``False`` keeps CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        self.lambda_init = float(lambda_init)
        self.use_en_frb = bool(use_en_frb)
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create FSREAdaTSKRegressor."""
        return FSREAdaTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGALETSKClassifierEstimator(FSREAdaTSKClassifierEstimator):
    """DG-ALETSK classifier with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-AdaTSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.

    Reference:
        G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional
        Fuzzy Approach With Simultaneous Feature Selection and Rule
        Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no.
        11, pp. 3866-3880, Nov. 2023, doi: 10.1109/TFUZZ.2023.3270445.

    Example:
        ```python
        from highfis import DGALETSKClassifierEstimator

        clf = DGALETSKClassifierEstimator(
            n_mfs=30, lambda_init=1.0, use_en_frb=False, random_state=0
        )
        clf.fit(X_train, y_train)
        ```
    """

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create DGALETSKClassifier."""
        return DGALETSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            lambda_init=self.lambda_init,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGALETSKRegressorEstimator(FSREAdaTSKRegressorEstimator):
    """DG-ALETSK regressor with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-AdaTSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.

    Reference:
        G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional
        Fuzzy Approach With Simultaneous Feature Selection and Rule
        Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no.
        11, pp. 3866-3880, Nov. 2023, doi: 10.1109/TFUZZ.2023.3270445.

    Example:
        ```python
        from highfis import DGALETSKRegressorEstimator

        reg = DGALETSKRegressorEstimator(
            n_mfs=30, lambda_init=1.0, use_en_frb=False, random_state=0
        )
        reg.fit(X_train, y_train)
        ```
    """

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create DGALETSKRegressor."""
        return DGALETSKRegressor(
            input_mfs,
            rule_base=rule_base,
            lambda_init=self.lambda_init,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGTSKClassifierEstimator(_BaseClassifierEstimator):
    """DG-TSK classifier with M-gate antecedent and point-based FRB (P-FRB).

    DG-TSK uses a data-guided M-gate function to automatically select
    relevant features and rules.

    Reference:
        Guangdong Xue, Jian Wang, Bingjie Zhang, Bin Yuan, Caili Dai,
        Double groups of gates based Takagi-Sugeno-Kang (DG-TSK)
        fuzzy system for simultaneous feature selection and rule
        extraction, Fuzzy Sets and Systems, Volume 469, 2023, 108627,
        ISSN 0165-0114, https://doi.org/10.1016/j.fss.2023.108627.

    Example:
        ```python
        from highfis import DGTSKClassifierEstimator

        clf = DGTSKClassifierEstimator(n_mfs=30, use_en_frb=False, random_state=0)
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a DG-TSK classifier.

        Args:
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) for rule
                extraction (P-FRB). Default ``False`` keeps CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum number of point-based FRB rules when
                ``rule_base='pfrb'``. ``None`` uses all training samples.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        self.use_en_frb = bool(use_en_frb)
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create DGTSKClassifier."""
        return DGTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGTSKRegressorEstimator(_BaseRegressorEstimator):
    """DG-TSK regressor with M-gate antecedent and point-based FRB (P-FRB).

    DG-TSK uses a data-guided M-gate function to automatically select
    relevant features and rules.

    Reference:
        Guangdong Xue, Jian Wang, Bingjie Zhang, Bin Yuan, Caili Dai,
        Double groups of gates based Takagi-Sugeno-Kang (DG-TSK)
        fuzzy system for simultaneous feature selection and rule
        extraction, Fuzzy Sets and Systems, Volume 469, 2023, 108627,
        ISSN 0165-0114, https://doi.org/10.1016/j.fss.2023.108627.

    Example:
        ```python
        from highfis import DGTSKRegressorEstimator

        reg = DGTSKRegressorEstimator(n_mfs=30, use_en_frb=False, random_state=0)
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a DG-TSK regressor.

        Args:
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) for rule
                extraction (P-FRB). Default ``False`` keeps CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum number of point-based FRB rules when
                ``rule_base='pfrb'``. ``None`` uses all training samples.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        self.use_en_frb = bool(use_en_frb)
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create DGTSKRegressor."""
        return DGTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


# =====================================================================
# LogTSK Estimators  (Cui, Wu & Xu, IEEE TFS 2021)
# =====================================================================


class LogTSKClassifierEstimator(_BaseClassifierEstimator):
    r"""LogTSK classifier with inverse-log rule normalization.

    LogTSK uses product antecedent aggregation and inverse-log
    normalization of log-domain rule strengths. The resulting
    rule weights are normalized with L1 normalization across
    rules, which makes the model scale-invariant in log-space
    and avoids the softmax saturation that occurs in
    high-dimensional inputs.

    Reference:
        Y. Cui, D. Wu and Y. Xu, "Curse of Dimensionality for TSK Fuzzy Neural
        Networks: Explanation and Solutions," 2021 International Joint
        Conference on Neural Networks (IJCNN), pp. 1-8,
        doi: 10.1109/IJCNN52387.2021.9534265.

    Example:
        ```python
        from highfis import LogTSKClassifierEstimator

        clf = LogTSKClassifierEstimator()
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a LogTSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` is recommended (the
                log-space defuzzifier is scale-invariant).
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create LogTSKClassifier."""
        return LogTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class LogTSKRegressorEstimator(_BaseRegressorEstimator):
    r"""LogTSK regressor with inverse-log rule normalization.

    LogTSK uses product antecedent aggregation and inverse-log
    normalization of log-domain rule strengths. The resulting
    rule weights are normalized with L1 normalization across
    rules, which makes the model scale-invariant in log-space
    and avoids the softmax saturation that occurs in
    high-dimensional inputs.

    Reference:
        Y. Cui, D. Wu and Y. Xu, "Curse of Dimensionality for TSK Fuzzy Neural
        Networks: Explanation and Solutions," 2021 International Joint
        Conference on Neural Networks (IJCNN), pp. 1-8,
        doi: 10.1109/IJCNN52387.2021.9534265.

    Example:
        ```python
        from highfis import LogTSKRegressorEstimator

        reg = LogTSKRegressorEstimator()
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a LogTSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` is recommended (the
                log-space defuzzifier is scale-invariant).
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create LogTSKRegressor."""
        return LogTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


__all__: list[str] = [
    "AdaTSKClassifierEstimator",
    "AdaTSKRegressorEstimator",
    "DGALETSKClassifierEstimator",
    "DGALETSKRegressorEstimator",
    "DGTSKClassifierEstimator",
    "DGTSKRegressorEstimator",
    "DombiTSKClassifierEstimator",
    "DombiTSKRegressorEstimator",
    "FSREAdaTSKClassifierEstimator",
    "FSREAdaTSKRegressorEstimator",
    "HTSKClassifierEstimator",
    "HTSKRegressorEstimator",
    "InputConfig",
    "LogTSKClassifierEstimator",
    "LogTSKRegressorEstimator",
    "TSKClassifierEstimator",
    "TSKRegressorEstimator",
    "_build_kmeans_input_mfs",
]
