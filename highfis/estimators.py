"""Scikit-learn compatible estimator wrappers for highFIS TSK models.

This module provides high-level, sklearn-compatible wrappers for every TSK
variant implemented in :mod:`highfis.models`. Each estimator follows the
standard ``fit`` / ``predict`` / ``score`` interface and handles membership-
function initialisation, model construction and the training loop internally.

Two abstract base classes share the common logic:

* :class:`_BaseClassifierEstimator` — for classification tasks.
* :class:`_BaseRegressorEstimator` — for regression tasks.

Concrete estimators cover the following model families:

* **TSK** — vanilla Takagi-Sugeno-Kang (Takagi & Sugeno, 1985).
* **HTSK** — high-dimensional TSK via averaged defuzzification
  (Cui et al., IJCNN 2021).
* **LogTSK** — log-transformed defuzzification for high-dimensional data
  (Du et al., 2020; analysed by Cui et al., IJCNN 2021).
* **DombiTSK** — Dombi T-norm based TSK (Xue et al., TFS 2025).
* **AYATSK** — adaptive Yager T-norm based TSK (Xue et al., TSMC 2025).
* **AdaTSK** — adaptive softmin based TSK (Xue et al., IJCNN 2022).
* **FSRE-AdaTSK** — AdaTSK with feature-selection and rule-extraction gates
  (Xue et al., TFS 2022).
* **DG-ALETSK** — double-gate adaptive Ln-Exp softmin TSK
  (Xue et al., TFS 2023).
* **DG-TSK** — double-gate TSK with point-based FRB
  (Xue et al., Fuzzy Sets and Systems, 2023).

Membership-function initialisation:

* ``mf_init="kmeans"`` (default) — k-means cluster centroids become MF
  centres; sigma is derived from within-cluster spread scaled by
  ``sigma_scale``. Produces a CoCo rule base by default.
* ``mf_init="grid"`` — regular-grid placement controlled by
  :class:`InputConfig`. Produces a Cartesian rule base by default.
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
from .memberships import GaussianMF
from .models import (
    AdaTSKClassifier,
    AdaTSKRegressor,
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
        >>> from highfis.estimators import InputConfig
        >>> configs = [
        ...     InputConfig(name="sepal_length", n_mfs=3),
        ...     InputConfig(name="sepal_width", n_mfs=5, overlap=0.3),
        ... ]
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
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int = 20,
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
            verbose: If ``True``, prints per-epoch loss and accuracy to
                stdout during :meth:`fit`.
            rule_base: Explicit rule-base construction type. ``"coco"``
                (compactly combined) pairs rule ``r`` with MF ``r`` on every
                feature. ``"cartesian"`` enumerates all MF combinations.
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
            input_mfs = _build_gaussian_input_mfs(x_arr, input_configs)
            feature_names = [cfg.name for cfg in input_configs]
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
            verbose=bool(self.verbose),
            x_val=x_val_t,
            y_val=y_val_t,
            patience=int(self.patience),
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
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
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
            verbose: If ``True``, prints per-epoch loss and metrics to
                stdout during :meth:`fit`.
            rule_base: Explicit rule-base construction type. ``"coco"``
                (compactly combined) pairs rule ``r`` with MF ``r`` on every
                feature. ``"cartesian"`` enumerates all MF combinations.
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
            patience: Number of consecutive epochs without improvement on
                the validation loss before training is stopped early. Only
                active when ``validation_data`` is provided.
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
        self.patience = patience
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
            input_mfs = _build_gaussian_input_mfs(x_arr, input_configs)
            feature_names = [cfg.name for cfg in input_configs]
            effective_rule_base = self.rule_base if self.rule_base is not None else "cartesian"

        return input_mfs, feature_names, effective_rule_base

    @abstractmethod
    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
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

        self.model_ = self._build_model(input_mfs, effective_rule_base)

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
            verbose=bool(self.verbose),
            x_val=x_val_t,
            y_val=y_val_t,
            patience=int(self.patience),
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
        estimator.model_ = estimator._build_model(
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


# =====================================================================
# HTSK Estimators
# =====================================================================


class HTSKClassifierEstimator(_BaseClassifierEstimator):
    """High-dimensional TSK classifier (Cui et al., IJCNN 2021).

    Wraps :class:`~highfis.models.HTSKClassifier`. HTSK replaces the standard
    softmax-based defuzzification with a D-th-root variant that eliminates
    saturation on high-dimensional inputs without inflating sigma. It is the
    recommended default TSK classifier for datasets with more than ~50
    features.

    The experimental setup from the original paper used ``n_mfs=30``,
    ``sigma_scale=1.0``, ``mf_init="kmeans"``, ``epochs=200``,
    ``learning_rate=0.01``, ``batch_size=512``, and ``patience=20``.

    Example:
        >>> from highfis import HTSKClassifierEstimator
        >>> clf = HTSKClassifierEstimator(n_mfs=30, random_state=0)
        >>> clf.fit(X_train, y_train)
        HTSKClassifierEstimator(...)
        >>> clf.score(X_test, y_test)
        0.95...
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an HTSK classifier.

        HTSK (Cui et al., IJCNN 2021) replaces the standard softmax
        defuzzification with a dimensionality-normalised variant that averages
        the exponent over all input features::

            f_r(x)^{1/D} / sum_i f_i(x)^{1/D}

        This prevents softmax saturation on high-dimensional data without
        requiring an inflated ``sigma_scale``. The recommended initialisation
        is ``sigma_scale=1.0`` regardless of dimensionality.

        Reference:
            Cui, Y., Wu, D., & Xu, Y. (2021). Curse of dimensionality for
            TSK fuzzy neural networks: Explanation and solutions. In *Proc.
            IJCNN*, pp. 1-8. https://doi.org/10.1109/IJCNN52387.2021.9534265

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs. Cui et al.
                (2021) used ``R=30``.
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` is recommended for
                HTSK because the defuzzifier already compensates for
                dimensionality.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
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
            patience: Early-stopping patience (default ``20``).
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
    """High-dimensional TSK regressor (Cui et al., IJCNN 2021).

    Wraps :class:`~highfis.models.HTSKRegressor`. HTSK replaces the standard
    softmax-based defuzzification with a D-th-root variant that eliminates
    saturation on high-dimensional inputs without inflating sigma. It is the
    recommended default TSK regressor for datasets with more than ~50
    features.

    The experimental setup from the original paper used ``n_mfs=30``,
    ``sigma_scale=1.0``, ``mf_init="kmeans"``, ``epochs=200``,
    ``learning_rate=0.01``, ``batch_size=512``, and ``patience=20``.

    Example:
        >>> from highfis import HTSKRegressorEstimator
        >>> reg = HTSKRegressorEstimator(n_mfs=30, random_state=0)
        >>> reg.fit(X_train, y_train)
        HTSKRegressorEstimator(...)
        >>> reg.score(X_test, y_test)
        0.87...
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an HTSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs.
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` is recommended because
                HTSK defuzzification already compensates for dimensionality.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
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
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
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
    """Vanilla TSK classifier with sum-based (softmax) defuzzification.

    Wraps :class:`~highfis.models.TSKClassifier`. Implements the original
    Takagi-Sugeno-Kang inference with product T-norm and centre-of-gravity
    defuzzification. On high-dimensional data (``D ≥ ~50``) the softmax
    normalisation saturates, causing most inputs to fire only one rule. In
    that scenario consider :class:`HTSKClassifierEstimator` or
    :class:`LogTSKClassifierEstimator`, or increase ``sigma_scale`` (``"auto"``
    sets it to ``sqrt(D)`` as analysed by Cui et al., IJCNN 2021).

    Example:
        >>> from highfis import TSKClassifierEstimator
        >>> clf = TSKClassifierEstimator(n_mfs=30, random_state=0)
        >>> clf.fit(X_train, y_train)
        TSKClassifierEstimator(...)
        >>> clf.score(X_test, y_test)
        0.91...
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a vanilla TSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. Use ``"auto"`` (= ``sqrt(D)``)
                for high-dimensional data to mitigate softmax saturation
                (Cui et al., IJCNN 2021). ``1.0`` is appropriate for low-
                to medium-dimensional problems.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
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
            patience: Early-stopping patience (default ``20``).
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
    """Vanilla TSK regressor with sum-based (softmax) defuzzification.

    Wraps :class:`~highfis.models.TSKRegressor`. Implements the original
    Takagi-Sugeno-Kang inference. For high-dimensional datasets consider
    :class:`HTSKRegressorEstimator` or :class:`LogTSKRegressorEstimator`.

    Example:
        >>> from highfis import TSKRegressorEstimator
        >>> reg = TSKRegressorEstimator(n_mfs=30, random_state=0)
        >>> reg.fit(X_train, y_train)
        TSKRegressorEstimator(...)
        >>> reg.score(X_test, y_test)
        0.85...
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a vanilla TSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. Use ``"auto"`` (= ``sqrt(D)``)
                for high-dimensional data to mitigate softmax saturation.
                ``1.0`` is appropriate for low-to-medium-dimensional problems.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
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
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
    ) -> BaseTSK:
        """Create TSKRegressor."""
        return TSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class AYATSKClassifierEstimator(_BaseClassifierEstimator):
    """Adaptive Yager T-norm TSK classifier (Xue et al., TSMC 2025).

    Wraps :class:`~highfis.models.AYATSKClassifier`. AYATSK uses the Yager
    T-norm with an adaptive index parameter ``λ`` derived from the input
    dimensionality ``D`` and the lower bound ``ε`` of the Composite
    Exponential Membership Function (CEMF)::

        λ = -ln(D) / ln(1 - ε)

    This guarantees that the Yager T-norm remains numerically stable on
    high-dimensional data (no underflow) while being a proper T-norm,
    unlike softmin-based approaches. CEMF has a positive lower bound
    ``1/K > 0``, required for the adaptive strategy to be well-defined.

    Experiments in Xue et al. (2025) cover datasets with dimensionality
    from 8 to 120 432, making AYATSK one of the most broadly validated
    high-dimensional fuzzy classifiers available.

    Example:
        >>> from highfis import AYATSKClassifierEstimator
        >>> clf = AYATSKClassifierEstimator(n_mfs=30, random_state=0)
        >>> clf.fit(X_train, y_train)
        AYATSKClassifierEstimator(...)
        >>> clf.score(X_test, y_test)
        0.94...
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an AYATSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor for k-means initialisation.
                ``1.0`` is recommended; the adaptive Yager T-norm handles
                high-dimensional stability internally.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
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
            patience: Early-stopping patience (default ``20``).
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
    """Adaptive Yager T-norm TSK regressor (Xue et al., TSMC 2025).

    Wraps :class:`~highfis.models.AYATSKRegressor`. See
    :class:`AYATSKClassifierEstimator` for a description of the AYATSK
    model.

    Example:
        >>> from highfis import AYATSKRegressorEstimator
        >>> reg = AYATSKRegressorEstimator(n_mfs=30, random_state=0)
        >>> reg.fit(X_train, y_train)
        AYATSKRegressorEstimator(...)
        >>> reg.score(X_test, y_test)
        0.88...
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an AYATSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
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
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
    ) -> BaseTSK:
        """Create AYATSKRegressor."""
        return AYATSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class DombiTSKClassifierEstimator(_BaseClassifierEstimator):
    """Dombi T-norm based TSK classifier (Xue et al., TFS 2025).

    Wraps :class:`~highfis.models.DombiTSKClassifier`. DombiTSK replaces the
    product T-norm with the Dombi T-norm::

        φ_r(x) = 1 / (1 + [Σ_d (1/μ_{r,d}(x) - 1)^λ]^{1/λ})

    Dombi T-norm is differentiable and does not trigger numeric underflow even
    for very high-dimensional inputs (Xue et al., TFS 2025, Table I shows no
    underflow up to ``D=120432``). The static ``DombiTSK`` variant uses a
    fixed ``λ``; see :class:`AdaTSKClassifierEstimator` for the adaptive
    version (**ADMTSK**) in which ``λ`` is set automatically from ``D`` and
    the membership lower bound.

    Reference:
        Xue, G., Hu, L., Wang, J., & Ablameyko, S. (2025). ADMTSK: A
        high-dimensional TSK fuzzy system based on adaptive Dombi T-norm.
        *IEEE Trans. Fuzzy Systems*.
        https://doi.org/10.1109/TFUZZ.2025.3535640

    Example:
        >>> from highfis import DombiTSKClassifierEstimator
        >>> clf = DombiTSKClassifierEstimator(n_mfs=30, random_state=0)
        >>> clf.fit(X_train, y_train)
        DombiTSKClassifierEstimator(...)
        >>> clf.score(X_test, y_test)
        0.93...
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a DombiTSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended; the Dombi
                T-norm handles high-dimensional stability without inflating
                sigma.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
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
            patience: Early-stopping patience (default ``20``).
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
    """Dombi T-norm based TSK regressor (Xue et al., TFS 2025).

    Wraps :class:`~highfis.models.DombiTSKRegressor`. See
    :class:`DombiTSKClassifierEstimator` for a description of the DombiTSK
    model.

    Example:
        >>> from highfis import DombiTSKRegressorEstimator
        >>> reg = DombiTSKRegressorEstimator(n_mfs=30, random_state=0)
        >>> reg.fit(X_train, y_train)
        DombiTSKRegressorEstimator(...)
        >>> reg.score(X_test, y_test)
        0.87...
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a DombiTSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
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
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
    ) -> BaseTSK:
        """Create DombiTSKRegressor."""
        return DombiTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class AdaTSKClassifierEstimator(_BaseClassifierEstimator):
    """Adaptive Dombi T-norm TSK classifier — ADMTSK (Xue et al., TFS 2025).

    Wraps :class:`~highfis.models.AdaTSKClassifier`. Extends
    :class:`DombiTSKClassifierEstimator` with an **adaptive** index parameter
    ``λ`` for the Dombi T-norm. The adaptive strategy sets ``λ`` from the
    input dimensionality ``D`` and the theoretical lower bound ``ε`` of the
    membership function, ensuring no numeric underflow regardless of ``D``.

    The extra hyperparameter ``lambda_init`` seeds the initial value of ``λ``
    before training begins.

    Reference:
        Xue, G., Hu, L., Wang, J., & Ablameyko, S. (2025). ADMTSK: A
        high-dimensional TSK fuzzy system based on adaptive Dombi T-norm.
        *IEEE Trans. Fuzzy Systems*.
        https://doi.org/10.1109/TFUZZ.2025.3535640

    Example:
        >>> from highfis import AdaTSKClassifierEstimator
        >>> clf = AdaTSKClassifierEstimator(n_mfs=30, lambda_init=1.0, random_state=0)
        >>> clf.fit(X_train, y_train)
        AdaTSKClassifierEstimator(...)
        >>> clf.score(X_test, y_test)
        0.95...
    """

    def __init__(
        self,
        *,
        lambda_init: float = 1.0,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an AdaTSK (ADMTSK) classifier.

        Args:
            lambda_init: Initial value of the Dombi T-norm index parameter
                ``λ > 0``. The adaptive strategy will adjust ``λ`` during
                the forward pass based on current membership values and
                dimensionality. Xue et al. (2025) used ``lambda_init=1.0``.
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended; the
                adaptive Dombi T-norm handles high-dimensional stability.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        self.lambda_init = float(lambda_init)
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
            lambda_init=self.lambda_init,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class AdaTSKRegressorEstimator(_BaseRegressorEstimator):
    """Adaptive Dombi T-norm TSK regressor — ADMTSK (Xue et al., TFS 2025).

    Wraps :class:`~highfis.models.AdaTSKRegressor`. See
    :class:`AdaTSKClassifierEstimator` for a description of the ADMTSK model.

    Example:
        >>> from highfis import AdaTSKRegressorEstimator
        >>> reg = AdaTSKRegressorEstimator(n_mfs=30, lambda_init=1.0, random_state=0)
        >>> reg.fit(X_train, y_train)
        AdaTSKRegressorEstimator(...)
        >>> reg.score(X_test, y_test)
        0.89...
    """

    def __init__(
        self,
        *,
        lambda_init: float = 1.0,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an AdaTSK (ADMTSK) regressor.

        Args:
            lambda_init: Initial Dombi T-norm index ``λ > 0``. Xue et al.
                (2025) used ``lambda_init=1.0``.
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
            validation_data: Optional ``(X_val, y_val)`` for early stopping.
            weight_decay: L2 weight decay for consequent parameters.

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        self.lambda_init = float(lambda_init)
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
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
    ) -> BaseTSK:
        """Create AdaTSKRegressor."""
        return AdaTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            lambda_init=self.lambda_init,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class FSREAdaTSKClassifierEstimator(_BaseClassifierEstimator):
    """FSRE-AdaTSK classifier with feature selection and rule extraction.

    Wraps :class:`~highfis.models.FSREAdaTSKClassifier`. FSRE-AdaTSK extends
    AdaTSK by embedding **gate functions** in both the antecedents (for
    feature selection, FS) and the consequents (for rule extraction, RE).
    Training is performed in two sequential phases:

    1. FS phase — feature gates are trained alongside all system parameters;
       low-gate features are pruned.
    2. RE phase — rule gates are trained after building an Enhanced FRB
       (En-FRB) from the remaining features; low-gate rules are pruned.

    When ``use_en_frb=True`` the Enhanced FRB (En-FRB), whose size grows
    linearly with the number of features, is used; the default ``False``
    keeps the standard CoCo-FRB.

    Reference:
        Xue, G., Wang, J., Yuan, B., & Dai, C. (2023). DG-ALETSK: A
        high-dimensional fuzzy approach with simultaneous feature selection
        and rule extraction. *IEEE Trans. Fuzzy Systems*, 31(11).
        https://doi.org/10.1109/TFUZZ.2023.3270445

    Example:
        >>> from highfis import FSREAdaTSKClassifierEstimator
        >>> clf = FSREAdaTSKClassifierEstimator(
        ...     n_mfs=30, lambda_init=1.0, use_en_frb=False, random_state=0
        ... )
        >>> clf.fit(X_train, y_train)
        FSREAdaTSKClassifierEstimator(...)
        >>> clf.score(X_test, y_test)
        0.94...
    """

    def __init__(
        self,
        *,
        lambda_init: float = 1.0,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an FSRE-AdaTSK classifier.

        Args:
            lambda_init: Initial adaptive softmin index parameter ``λ > 0``.
                Controls the initial approximation quality to the minimum
                T-norm. Xue et al. (2023) used ``lambda_init=1.0``.
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) whose
                size grows linearly with the number of features, allowing
                more candidate rules for the RE phase. Xue et al. (2023)
                activate En-FRB after the FS phase; set ``False`` (default)
                to keep the compact CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
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
            lambda_init=self.lambda_init,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class FSREAdaTSKRegressorEstimator(_BaseRegressorEstimator):
    """FSRE-AdaTSK regressor with feature selection and rule extraction.

    Wraps :class:`~highfis.models.FSREAdaTSKRegressor`. See
    :class:`FSREAdaTSKClassifierEstimator` for a description of the
    FSRE-AdaTSK model.

    Example:
        >>> from highfis import FSREAdaTSKRegressorEstimator
        >>> reg = FSREAdaTSKRegressorEstimator(
        ...     n_mfs=30, lambda_init=1.0, use_en_frb=False, random_state=0
        ... )
        >>> reg.fit(X_train, y_train)
        FSREAdaTSKRegressorEstimator(...)
        >>> reg.score(X_test, y_test)
        0.88...
    """

    def __init__(
        self,
        *,
        lambda_init: float = 1.0,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an FSRE-AdaTSK regressor.

        Args:
            lambda_init: Initial adaptive softmin index ``λ > 0``. Xue et al.
                (2023) used ``lambda_init=1.0``.
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) for rule
                extraction. Default ``False`` keeps CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
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
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
    ) -> BaseTSK:
        """Create FSREAdaTSKRegressor."""
        return FSREAdaTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            lambda_init=self.lambda_init,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGALETSKClassifierEstimator(FSREAdaTSKClassifierEstimator):
    """DG-ALETSK classifier with ALE-softmin firing strength.

    Wraps :class:`~highfis.models.DGALETSKClassifier`. DG-ALETSK (Xue et al.,
    IEEE TFUZZ 2023, https://doi.org/10.1109/TFUZZ.2023.3270445) extends
    FSRE-AdaTSK by replacing the adaptive softmin with the *Adaptive
    Ln-Exp (ALE)* softmin, a smoother and more numerically stable variant.

    All constructor parameters are identical to
    :class:`FSREAdaTSKClassifierEstimator`.

    Example:
        >>> from highfis import DGALETSKClassifierEstimator
        >>> clf = DGALETSKClassifierEstimator(
        ...     n_mfs=30, lambda_init=1.0, use_en_frb=False, random_state=0
        ... )
        >>> clf.fit(X_train, y_train)
        DGALETSKClassifierEstimator(...)
        >>> clf.score(X_test, y_test)
        0.91...
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
    """DG-ALETSK regressor with ALE-softmin firing strength.

    Wraps :class:`~highfis.models.DGALETSKRegressor`. See
    :class:`DGALETSKClassifierEstimator` for a description of the model.

    All constructor parameters are identical to
    :class:`FSREAdaTSKRegressorEstimator`.

    Example:
        >>> from highfis import DGALETSKRegressorEstimator
        >>> reg = DGALETSKRegressorEstimator(
        ...     n_mfs=30, lambda_init=1.0, use_en_frb=False, random_state=0
        ... )
        >>> reg.fit(X_train, y_train)
        DGALETSKRegressorEstimator(...)
        >>> reg.score(X_test, y_test)
        0.88...
    """

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
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
    """DG-TSK classifier with M-gate and point-based FRB (P-FRB).

    Wraps :class:`~highfis.models.DGTSKClassifier`. DG-TSK (Xue et al.,
    Fuzzy Sets and Systems 2023, https://doi.org/10.1016/j.fss.2023.108627)
    introduces a *data-driven gate* (M-gate) to automatically select relevant
    rules and supports a *point-based FRB* (P-FRB) for compact rule sets.

    Example:
        >>> from highfis import DGTSKClassifierEstimator
        >>> clf = DGTSKClassifierEstimator(n_mfs=30, use_en_frb=False, random_state=0)
        >>> clf.fit(X_train, y_train)
        DGTSKClassifierEstimator(...)
        >>> clf.score(X_test, y_test)
        0.92...
    """

    def __init__(
        self,
        *,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a DG-TSK classifier.

        Args:
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) for rule
                extraction (P-FRB). Default ``False`` keeps CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
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
            patience=patience,
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
    """DG-TSK regressor with M-gate and point-based FRB (P-FRB).

    Wraps :class:`~highfis.models.DGTSKRegressor`. See
    :class:`DGTSKClassifierEstimator` for a description of the model.

    Example:
        >>> from highfis import DGTSKRegressorEstimator
        >>> reg = DGTSKRegressorEstimator(n_mfs=30, use_en_frb=False, random_state=0)
        >>> reg.fit(X_train, y_train)
        DGTSKRegressorEstimator(...)
        >>> reg.score(X_test, y_test)
        0.88...
    """

    def __init__(
        self,
        *,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a DG-TSK regressor.

        Args:
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) for rule
                extraction (P-FRB). Default ``False`` keeps CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
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
            patience=patience,
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
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
    """LogTSK classifier with log-space defuzzification.

    Wraps :class:`~highfis.models.LogTSKClassifier`. LogTSK (Du et al., 2020;
    analysed by Cui et al., IJCNN 2021, https://doi.org/10.1109/IJCNN52387.2021.9534099)
    computes firing strengths in log-space to improve numerical stability.
    L1 normalisation makes the model scale-invariant (Section III-F of Cui et
    al.), so ``sigma_scale=1.0`` is the recommended default.

    Example:
        >>> from highfis import LogTSKClassifierEstimator
        >>> clf = LogTSKClassifierEstimator(n_mfs=30, random_state=0)
        >>> clf.fit(X_train, y_train)
        LogTSKClassifierEstimator(...)
        >>> clf.score(X_test, y_test)
        0.90...
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a LogTSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` is recommended (the
                log-space defuzzifier is scale-invariant).
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
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
    """LogTSK regressor with log-space defuzzification.

    Wraps :class:`~highfis.models.LogTSKRegressor`. See
    :class:`LogTSKClassifierEstimator` for a description of the model.

    Example:
        >>> from highfis import LogTSKRegressorEstimator
        >>> reg = LogTSKRegressorEstimator(n_mfs=30, random_state=0)
        >>> reg.fit(X_train, y_train)
        LogTSKRegressorEstimator(...)
        >>> reg.score(X_test, y_test)
        0.88...
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 30,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        verbose: bool = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int = 20,
        validation_data: tuple[Any, Any] | None = None,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a LogTSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``30``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` is recommended (the
                log-space defuzzifier is scale-invariant).
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
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
            validation_data=validation_data,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: dict[str, list[GaussianMF]],
        rule_base: str,
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
