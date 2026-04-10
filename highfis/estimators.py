from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .memberships import GaussianMF
from .models import HTSKClassifier, HTSKRegressor


@dataclass(frozen=True)
class InputConfig:
    """Input configuration for Gaussian MF initialization."""

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


class HTSKClassifierEstimator(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """High-level HTSK classifier facade with sklearn-compatible API."""

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
        r"""Configure estimator hyperparameters and training options.

        Parameters
        ----------
        mf_init:
            Membership-function initialization strategy.  ``"kmeans"`` (default)
            places each rule's MF centers at a k-means centroid and uses the
            within-cluster standard deviation scaled by *sigma_scale* as the
            initial sigma — following Cui et al. (IJCNN 2021).  ``"grid"``
            uses a uniform grid over each feature's range (original behaviour,
            combined with ``rule_base="cartesian"``).
        sigma_scale:
            Scaling factor ``h`` applied to the computed sigma values during
            k-means initialization.  The paper recommends ``h=1`` for HTSK.
            Pass ``"auto"`` to use :math:`h = \\sqrt{D}` where *D* is the
            number of input features — a robust choice for any dimensionality.
            Ignored when ``mf_init="grid"``.
        rule_base:
            Rule-combination strategy passed to :class:`HTSKClassifier`.
            Defaults to ``"coco"`` when ``mf_init="kmeans"`` (one rule per
            cluster) and to ``"cartesian"`` when ``mf_init="grid"``.
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

    def _resolve_input_configs(self, x: np.ndarray) -> list[InputConfig]:
        if self.input_configs is not None:
            if len(self.input_configs) != x.shape[1]:
                raise ValueError(
                    f"input_configs length ({len(self.input_configs)}) must match number of features ({x.shape[1]})"
                )
            return list(self.input_configs)
        return [InputConfig(name=f"x{i + 1}", n_mfs=int(self.n_mfs)) for i in range(x.shape[1])]

    def _resolve_feature_names(self, x: np.ndarray) -> list[str]:
        if self.input_configs is not None:
            if len(self.input_configs) != x.shape[1]:
                raise ValueError(
                    f"input_configs length ({len(self.input_configs)}) must match number of features ({x.shape[1]})"
                )
            return [cfg.name for cfg in self.input_configs]
        return [f"x{i + 1}" for i in range(x.shape[1])]

    @staticmethod
    def _as_tensor_x(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32)

    def fit(self, x: Any, y: Any) -> HTSKClassifierEstimator:
        """Train the HTSK classifier on labeled samples."""
        x_arr, y_arr = check_X_y(x, y)

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        le = LabelEncoder()
        y_idx = le.fit_transform(np.asarray(y_arr))

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

        self.n_features_in_ = x_arr.shape[1]
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.classes_ = le.classes_
        self._label_encoder_ = le

        self.model_ = HTSKClassifier(
            input_mfs,
            n_classes=len(self.classes_),
            rule_base=effective_rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )

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
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        """Predict class probabilities for input samples."""
        check_is_fitted(self, "model_")
        x_arr = check_array(x)
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_arr.shape[1]}")
        probs = self.model_.predict_proba(self._as_tensor_x(x_arr))
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


class HTSKRegressorEstimator(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """High-level HTSK regressor facade with sklearn-compatible API."""

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
        r"""Configure estimator hyperparameters and training options.

        Parameters
        ----------
        mf_init:
            Membership-function initialization strategy.  ``"kmeans"`` (default)
            places each rule's MF centers at a k-means centroid and uses the
            within-cluster standard deviation scaled by *sigma_scale* as the
            initial sigma — following Cui et al. (IJCNN 2021).  ``"grid"``
            uses a uniform grid over each feature's range (original behaviour,
            combined with ``rule_base="cartesian"``).
        sigma_scale:
            Scaling factor ``h`` applied to the computed sigma values during
            k-means initialization.  The paper recommends ``h=1`` for HTSK.
            Pass ``"auto"`` to use :math:`h = \\sqrt{D}` where *D* is the
            number of input features — a robust choice for any dimensionality.
            Ignored when ``mf_init="grid"``.
        rule_base:
            Rule-combination strategy passed to :class:`HTSKRegressor`.
            Defaults to ``"coco"`` when ``mf_init="kmeans"`` (one rule per
            cluster) and to ``"cartesian"`` when ``mf_init="grid"``.
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

    def _resolve_input_configs(self, x: np.ndarray) -> list[InputConfig]:
        if self.input_configs is not None:
            if len(self.input_configs) != x.shape[1]:
                raise ValueError(
                    f"input_configs length ({len(self.input_configs)}) must match number of features ({x.shape[1]})"
                )
            return list(self.input_configs)
        return [InputConfig(name=f"x{i + 1}", n_mfs=int(self.n_mfs)) for i in range(x.shape[1])]

    def _resolve_feature_names(self, x: np.ndarray) -> list[str]:
        if self.input_configs is not None:
            if len(self.input_configs) != x.shape[1]:
                raise ValueError(
                    f"input_configs length ({len(self.input_configs)}) must match number of features ({x.shape[1]})"
                )
            return [cfg.name for cfg in self.input_configs]
        return [f"x{i + 1}" for i in range(x.shape[1])]

    @staticmethod
    def _as_tensor_x(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32)

    def fit(self, x: Any, y: Any) -> HTSKRegressorEstimator:
        """Train the HTSK regressor on labeled samples."""
        x_arr, y_arr = check_X_y(x, y)

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

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

        self.n_features_in_ = x_arr.shape[1]
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)

        self.model_ = HTSKRegressor(
            input_mfs,
            rule_base=effective_rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )

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
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Predict continuous target values for input samples."""
        check_is_fitted(self, "model_")
        x_arr = check_array(x)
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_arr.shape[1]}")
        preds = self.model_.predict(self._as_tensor_x(x_arr))
        return preds.detach().cpu().numpy()


__all__ = [
    "InputConfig",
    "HTSKClassifierEstimator",
    "HTSKRegressorEstimator",
    "_build_kmeans_input_mfs",
]
