"""Base estimator classes, input configuration, and shared utilities."""

from __future__ import annotations

import math
import os
import threading
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Self, cast

import numpy as np
import numpy.typing as npt
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import Tensor

from ..clustering import FuzzyCMeans, KMeans, MiniBatchKMeans
from ..memberships import (
    DimensionDependentGaussianMF,
    GaussianMF,
    GaussianPiMF,
    MembershipFunction,
)
from ..metrics import compute_metrics
from ..models import BaseTSK
from ..models._base import set_training_flag
from ..optim._base import BaseTrainer
from ..optim._gradient import GradientTrainer
from ..persistence import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    deserialize_input_mfs,
    load_checkpoint,
    save_checkpoint,
    serialize_input_mfs,
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
    clusterer: KMeans | MiniBatchKMeans,
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
    clusterer.fit(x)
    if clusterer.cluster_centers_ is None:
        raise RuntimeError("KMeans did not compute cluster centers")

    if hasattr(clusterer.cluster_centers_, "cpu"):
        centers = clusterer.cluster_centers_.cpu().numpy()
    else:
        centers = np.asarray(clusterer.cluster_centers_)

    labels: np.ndarray = np.asarray(clusterer.labels_)
    n_clusters = int(clusterer.n_clusters)
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


def _build_fuzzy_c_means_input_mfs(
    x: np.ndarray,
    clusterer: FuzzyCMeans,
    sigma_scale: float,
    feature_names: list[str],
    random_state: int | None,
) -> dict[str, list[GaussianMF]]:
    r"""Build Gaussian MFs via fuzzy C-means cluster initialization.

    The MF means are placed at the FCM centroids. Each sigma is sampled
    from ``N(h, 0.2)`` where ``h`` is the cluster-specific, feature-wise
    spread scaled by ``sigma_scale``.
    """
    clusterer.fit(x)
    if clusterer.cluster_centers_ is None or clusterer.membership_ is None:
        raise RuntimeError("FuzzyCMeans did not converge to a valid solution")

    centers: np.ndarray = clusterer.cluster_centers_.cpu().numpy()
    membership: np.ndarray = clusterer.membership_.cpu().numpy()
    n_clusters = int(clusterer.n_clusters)
    input_mfs: dict[str, list[GaussianMF]] = {}
    rng = np.random.default_rng(random_state)

    for d, name in enumerate(feature_names):
        col = x[:, d]
        center_col = centers[:, d]
        mfs: list[GaussianMF] = []
        for r in range(n_clusters):
            c = float(center_col[r])
            weights = membership[:, r] ** float(clusterer.m)
            total_weight = float(np.sum(weights))
            if total_weight > 0.0:
                variance = float(np.sum(weights * (col - c) ** 2) / total_weight)
                raw_sigma = float(np.sqrt(max(variance, 0.0)))
            else:
                raw_sigma = 0.0

            if raw_sigma < 1e-6:
                other = np.delete(center_col, r)
                raw_sigma = float(np.min(np.abs(other - c))) / 2.0 if len(other) > 0 else 1.0
            h = raw_sigma * sigma_scale
            sigma = max(float(rng.normal(loc=h, scale=0.2)), 1e-3)
            mfs.append(GaussianMF(mean=c, sigma=sigma))

        input_mfs[name] = mfs

    return input_mfs


def _resolve_clusterer(
    mf_init: str | KMeans | MiniBatchKMeans | FuzzyCMeans,
    n_clusters: int,
    random_state: int | None,
) -> KMeans | MiniBatchKMeans | FuzzyCMeans:
    """Resolve *mf_init* to a configured clustering object.

    When *mf_init* is already a clustering instance its ``n_clusters`` and
    ``random_state`` are overridden with the estimator values so that the
    number of rules and reproducibility are always controlled by the
    estimator's own parameters.
    """
    import copy

    if isinstance(mf_init, (KMeans, MiniBatchKMeans, FuzzyCMeans)):
        c = copy.copy(mf_init)
        c.n_clusters = n_clusters
        c.random_state = random_state
        return c
    init_str = mf_init.lower()
    if init_str == "kmeans":
        return KMeans(n_clusters=n_clusters, n_init=1, max_iter=100, random_state=random_state)
    if init_str == "minibatch_kmeans":
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    if init_str == "fcm":
        return FuzzyCMeans(n_clusters=n_clusters, random_state=random_state)
    raise ValueError(
        f"mf_init must be 'kmeans', 'minibatch_kmeans', 'fcm', 'grid', "
        f"or a KMeans/MiniBatchKMeans/FuzzyCMeans instance; got {mf_init!r}"
    )


def _select_pfrb_indices(
    n_samples: int,
    max_rules: int | None,
    random_state: int | None,
) -> np.ndarray:
    """Select the training-sample indices used to build a point-based FRB.

    Deterministic in ``(n_samples, max_rules, random_state)`` so the same
    sample subset can be reproduced when initialising the consequents from the
    corresponding labels (see :meth:`_BaseTSKEstimator._pfrb_aligned_labels`).
    When ``max_rules`` is ``None`` or covers every sample, all samples are used.
    """
    if max_rules is None or int(max_rules) >= n_samples:
        return np.arange(n_samples)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_samples, size=int(max_rules), replace=False)
    return np.sort(indices)


def _build_pfrb_input_mfs(
    x: np.ndarray,
    feature_names: list[str],
    max_rules: int | None,
    sigma_scale: float,
    random_state: int | None,
) -> dict[str, list[GaussianMF]]:
    """Build point-based fuzzy rule base membership functions from training samples."""
    indices = _select_pfrb_indices(x.shape[0], max_rules, random_state)

    input_mfs: dict[str, list[GaussianMF]] = {}
    for d, name in enumerate(feature_names):
        col = x[:, d]
        sigma = max(float(np.std(col)) * sigma_scale, 1e-3)
        centers = col[indices]
        input_mfs[name] = [GaussianMF(mean=float(c), sigma=sigma) for c in centers]
    return input_mfs


def _to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _normalize_importance(values: Tensor) -> np.ndarray:
    importance = _to_numpy(values)
    total = float(np.sum(importance))
    if total <= 0.0:
        return np.full(importance.shape, 1.0 / float(importance.shape[0]), dtype=np.float64)
    return importance / total


def _wrap_dimension_dependent_gaussian_input_mfs(
    input_mfs: Mapping[str, Sequence[MembershipFunction]],
    dimension: int,
    xi: float = 745.0,
    rho: float | None = None,
) -> dict[str, list[GaussianMF]]:
    return {
        name: cast(
            list[GaussianMF],
            [
                DimensionDependentGaussianMF(
                    mean=cast(GaussianMF, mf).mean.detach().item(),
                    sigma=cast(GaussianMF, mf).sigma.detach().item(),
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
    input_mfs: Mapping[str, Sequence[MembershipFunction]],
    eps: float | None = None,
) -> dict[str, list[GaussianMF]]:
    return {
        name: cast(
            list[GaussianMF],
            [
                GaussianPiMF(
                    mean=cast(GaussianMF, mf).mean.detach().item(),
                    sigma=cast(GaussianMF, mf).sigma.detach().item(),
                    k=1.0,
                    eps=eps if eps is not None else mf.eps,
                )
                for mf in mfs
            ],
        )
        for name, mfs in input_mfs.items()
    }


def _wrap_gaussian_pimf_input_mfs(
    input_mfs: Mapping[str, Sequence[MembershipFunction]],
    k: float = 1.0,
    eps: float | None = None,
) -> dict[str, list[GaussianMF]]:
    return {
        name: cast(
            list[GaussianMF],
            [
                GaussianPiMF(
                    mean=cast(GaussianMF, mf).mean.detach().item(),
                    sigma=cast(GaussianMF, mf).sigma.detach().item(),
                    k=k,
                    eps=eps if eps is not None else mf.eps,
                )
                for mf in mfs
            ],
        )
        for name, mfs in input_mfs.items()
    }


_DEFAULT_MF_CACHE_SIZE = 128
_MFCacheValue = tuple[dict[str, Any], list[str], str]


class MFCacheInfo(NamedTuple):
    """Snapshot of the membership-function initialisation cache state.

    Mirrors :meth:`functools.lru_cache().cache_info` with an extra ``enabled``
    flag. Returned by :func:`mf_cache_info`.
    """

    hits: int
    misses: int
    maxsize: int
    currsize: int
    enabled: bool


class _MFInitCache:
    """Thread-safe LRU cache for membership-function initialisation results.

    Keyed by :func:`_get_mf_cache_key`; values are the serialized MF config,
    feature names and effective rule base. Used to avoid recomputing k-means /
    grid initialisation on repeated ``fit`` calls with the same data and
    hyperparameters. Least-recently-used entries are evicted first (a cache hit
    renews the entry). Managed via the public helpers :func:`clear_mf_cache`,
    :func:`mf_cache_info`, :func:`set_mf_cache_enabled` and
    :func:`set_mf_cache_size`.
    """

    def __init__(self, maxsize: int = _DEFAULT_MF_CACHE_SIZE, enabled: bool = True) -> None:
        self._store: OrderedDict[Hashable, _MFCacheValue] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = max(1, int(maxsize))
        self._enabled = bool(enabled)
        self._hits = 0
        self._misses = 0

    def get(self, key: Hashable) -> _MFCacheValue | None:
        with self._lock:
            if not self._enabled or key not in self._store:
                self._misses += 1
                return None
            self._store.move_to_end(key)  # LRU: renew on access
            self._hits += 1
            return self._store[key]

    def set(self, key: Hashable, value: _MFCacheValue) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def info(self) -> MFCacheInfo:
        with self._lock:
            return MFCacheInfo(self._hits, self._misses, self._maxsize, len(self._store), self._enabled)

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def set_maxsize(self, maxsize: int) -> None:
        if int(maxsize) < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize!r}")
        with self._lock:
            self._maxsize = int(maxsize)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)


def _read_cache_env() -> dict[str, Any]:
    """Read cache configuration from environment variables (at import time).

    - ``HIGHFIS_DISABLE_MF_CACHE`` (truthy: ``1/true/yes/on``) disables the cache.
    - ``HIGHFIS_MF_CACHE_SIZE`` (positive int) sets the maximum number of entries;
      invalid or missing values fall back to the default.
    """
    disable = str(os.environ.get("HIGHFIS_DISABLE_MF_CACHE", "")).strip().lower()
    enabled = disable not in {"1", "true", "yes", "on"}

    maxsize = _DEFAULT_MF_CACHE_SIZE
    raw_size = os.environ.get("HIGHFIS_MF_CACHE_SIZE")
    if raw_size is not None:
        try:
            parsed = int(raw_size)
            if parsed >= 1:
                maxsize = parsed
        except (TypeError, ValueError):
            pass  # keep the default on invalid input

    return {"maxsize": maxsize, "enabled": enabled}


_MF_INIT_CACHE = _MFInitCache(**_read_cache_env())


def clear_mf_cache() -> None:
    """Clear the membership-function initialisation cache and reset its stats."""
    _MF_INIT_CACHE.clear()


def mf_cache_info() -> MFCacheInfo:
    """Return an :class:`MFCacheInfo` snapshot (hits, misses, maxsize, currsize, enabled)."""
    return _MF_INIT_CACHE.info()


def set_mf_cache_enabled(enabled: bool) -> None:
    """Enable or disable the membership-function initialisation cache.

    When disabled, every ``fit`` rebuilds the MFs and nothing is stored.
    """
    _MF_INIT_CACHE.set_enabled(enabled)


def set_mf_cache_size(maxsize: int) -> None:
    """Set the maximum number of cache entries (``>= 1``), evicting LRU entries if needed."""
    _MF_INIT_CACHE.set_maxsize(maxsize)


def _get_mf_cache_key(
    x_arr: np.ndarray,
    mf_init: Any,
    n_mfs: int,
    sigma_scale: Any,
    random_state: Any,
    pfrb_max_rules: Any,
    input_configs: list[InputConfig] | None,
    rule_base: Any = None,
) -> tuple[Any, ...]:
    # Determine step for sampling to hash quickly
    step = max(1, x_arr.shape[0] // 1000)
    sample_bytes = x_arr[::step].tobytes()
    data_hash = hash((x_arr.shape, x_arr.dtype, sample_bytes))

    # Normalize mf_init to a hashable type
    if isinstance(mf_init, str):
        mf_init_key: Any = mf_init.lower()
    elif mf_init is None:
        mf_init_key = None
    else:
        mf_init_key = (
            type(mf_init).__name__,
            getattr(mf_init, "n_clusters", None),
            getattr(mf_init, "random_state", None),
        )

    # Normalize input_configs to a hashable tuple
    input_configs_key = tuple(input_configs) if input_configs is not None else None

    return (
        data_hash,
        mf_init_key,
        n_mfs,
        sigma_scale,
        random_state,
        pfrb_max_rules,
        input_configs_key,
        rule_base,
    )


def _build_input_mfs_cached(
    estimator: Any,
    x_arr: np.ndarray,
    build_func: Callable[[np.ndarray], tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]],
) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
    cache_key = _get_mf_cache_key(
        x_arr,
        estimator.mf_init,
        estimator.n_mfs,
        estimator.sigma_scale,
        estimator.random_state,
        estimator.pfrb_max_rules,
        estimator.input_configs,
        getattr(estimator, "rule_base", None),
    )

    cached = _MF_INIT_CACHE.get(cache_key)
    if cached is not None:
        serialized_config, feature_names, effective_rule_base = cached
        # Reconstruct new MF objects from serialized parameters to ensure separate memory space/gradients
        return deserialize_input_mfs(serialized_config), feature_names, effective_rule_base

    # Otherwise, execute the real build function (outside the cache lock so the
    # expensive initialisation is not serialised across threads).
    input_mfs, feature_names, effective_rule_base = build_func(x_arr)

    serialized_config = serialize_input_mfs(input_mfs)
    _MF_INIT_CACHE.set(cache_key, (serialized_config, feature_names, effective_rule_base))

    # Reconstruct from serialized_config to match cached hits precisely.
    return deserialize_input_mfs(serialized_config), feature_names, effective_rule_base


class _BaseTSKEstimator(BaseEstimator):
    """Abstract parent class for TSK estimators in highFIS."""

    model_: BaseTSK
    feature_names_in_: np.ndarray | None
    rule_base_: str
    n_features_in_: int
    history_: dict[str, Any]

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str | KMeans | MiniBatchKMeans | FuzzyCMeans = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 100,
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
        weight_decay: float = 1e-8,
        device: str = "cpu",
        eval_metrics_every: int = 1,
        scheduler_class: type[Any] | None = None,
        scheduler_params: Mapping[str, Any] | None = None,
        trainer: BaseTrainer | None = None,
    ) -> None:
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
        self.weight_decay = weight_decay
        self.device = device
        self.eval_metrics_every = eval_metrics_every
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params
        self.trainer = trainer

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
    def _as_tensor_x(x: np.ndarray, device: torch.device | str | None = None) -> torch.Tensor:
        """Convert numpy array to a float32 tensor on *device*.

        Args:
            x: Input array to convert.
            device: Target PyTorch device. ``None`` uses the PyTorch default
                (CPU). Pass the resolved ``torch.device`` from :meth:`fit`
                or inference methods to keep all tensors on the same device
                as the model.
        """
        if not x.flags.writeable:
            x = x.copy()
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        """Build MFs and resolve rule_base from the initialization mode."""
        return _build_input_mfs_cached(self, x_arr, self._build_input_mfs_impl)

    def _build_input_mfs_impl(
        self, x_arr: np.ndarray
    ) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        # ---- grid initialisation ----
        if isinstance(self.mf_init, str) and self.mf_init.lower() == "grid":
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

        # ---- clustering-based initialisation ----
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
            clusterer = _resolve_clusterer(self.mf_init, int(self.n_mfs), self.random_state)
            if isinstance(clusterer, FuzzyCMeans):
                input_mfs = _build_fuzzy_c_means_input_mfs(
                    x_arr, clusterer, effective_sigma_scale, feature_names, self.random_state
                )
            else:
                input_mfs = _build_kmeans_input_mfs(
                    x_arr, clusterer, effective_sigma_scale, feature_names, self.random_state
                )
            effective_rule_base = self.rule_base if self.rule_base is not None else "coco"

        return input_mfs, feature_names, effective_rule_base

    def _get_trainer(self) -> BaseTrainer:
        """Return the default :class:`~highfis.optim.GradientTrainer` for this estimator.

        Subclasses may override this to return a different trainer, e.g.
        :class:`~highfis.optim.DGTrainer` for DG-TSK / DG-ALETSK estimators.
        """
        return GradientTrainer(
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            batch_size=self.batch_size,
            shuffle=bool(self.shuffle),
            patience=self.patience,
            restore_best=bool(self.restore_best),
            weight_decay=float(self.weight_decay),
            ur_weight=float(self.ur_weight),
            ur_target=self.ur_target,
            verbose=self.verbose,
            eval_metrics_every=self.eval_metrics_every,
            scheduler_class=self.scheduler_class,
            scheduler_params=self.scheduler_params,
        )

    def _pre_train_hook(self, model: BaseTSK, x_t: Tensor, y_t: Tensor) -> None:
        """Called just before the trainer runs.  No-op by default.

        Subclasses may override this to perform model-specific setup that
        depends on the already-built model and training tensors, e.g.
        P-FRB consequent initialisation in DG-TSK.
        """

    def _effective_pfrb_max_rules(self, n_features: int) -> int | None:
        """Return the P-FRB rule cap actually used to build the MFs.

        Defaults to ``pfrb_max_rules``. Subclasses that resolve a data-dependent
        default (e.g. DG-ALETSK's paper-style cap) must override this so that the
        consequent label initialisation samples the *same* points as the rule
        centres (see :meth:`_pfrb_aligned_labels`).
        """
        return self.pfrb_max_rules

    def _pfrb_aligned_labels(self, x_t: Tensor, y_t: Tensor) -> Tensor:
        """Return the labels of the sampled P-FRB points, aligned with rule centres.

        The P-FRB rule ``r`` is built from training sample ``indices[r]`` (see
        :func:`_build_pfrb_input_mfs`). This returns ``y_t[indices]`` so the
        one-hot consequent of rule ``r`` encodes the label of the *same* sample,
        rather than the ``r``-th label of the (unsampled) training set.
        """
        indices = _select_pfrb_indices(
            int(x_t.shape[0]),
            self._effective_pfrb_max_rules(int(x_t.shape[1])),
            self.random_state,
        )
        return y_t[torch.as_tensor(indices, dtype=torch.long, device=y_t.device)]

    @staticmethod
    def _model_is_first_order(model: Any) -> bool:
        """Whether *model* has been converted to a first-order gated consequent.

        Only models with a zero-order -> first-order conversion (DG-TSK, DG-ALETSK) can be
        in this state; the conversion swaps the consequent-layer class, so the persisted
        structure must be rebuilt on load before the weights fit.
        """
        return hasattr(model, "convert_to_first_order") and "ZeroOrder" not in type(model.consequent_layer).__name__

    @staticmethod
    def _restore_consequent_structure(model: Any, model_init: Mapping[str, Any]) -> None:
        """Rebuild a persisted gated consequent before loading its state dict.

        Mirrors the DG-TSK loader for every estimator that goes through the base ``load``:
        first re-apply the zero-order -> first-order conversion (so the parameter shapes
        match the checkpoint), then restore the consequent gate ``mode``.
        """
        if model_init.get("is_first_order", False) and hasattr(model, "convert_to_first_order"):
            cast(Any, model).convert_to_first_order()
        consequent_mode = model_init.get("consequent_mode")
        if consequent_mode is not None:
            if hasattr(model, "set_consequent_mode"):
                cast(Any, model).set_consequent_mode(consequent_mode)
            elif hasattr(model.consequent_layer, "mode"):
                model.consequent_layer.mode = consequent_mode

    def _build_checkpoint_base(
        self,
        *,
        model_init: dict[str, Any],
        fitted_attrs: dict[str, Any],
    ) -> dict[str, Any]:
        check_is_fitted(self, "model_")
        params = self.get_params(deep=False)
        if params.get("input_configs") is not None:
            params["input_configs"] = [
                {"name": c.name, "n_mfs": c.n_mfs, "overlap": c.overlap, "margin": c.margin}
                for c in params["input_configs"]
            ]
        # Exclude non-serialisable trainer objects from the checkpoint; they
        # are reconstructed from the estimator's hyperparameters on load.
        params.pop("trainer", None)
        # ``scheduler_class`` holds a class object, which ``torch.load(weights_only=True)``
        # refuses to unpickle -- keeping it would make the checkpoint unloadable. Its
        # partner goes with it: a schedule with no class to build is meaningless. Both are
        # training-time settings, and a reloaded estimator is not mid-training.
        params.pop("scheduler_class", None)
        params.pop("scheduler_params", None)
        return {
            "format": CHECKPOINT_FORMAT,
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "estimator_class": self.__class__.__name__,
            "estimator_params": params,
            "model_init": model_init,
            "model_state_dict": self.model_.state_dict(),
            "fitted_attrs": fitted_attrs,
            "history": getattr(self, "history_", None),
        }

    def get_mf_params(self) -> dict[str, list[dict[str, Any]]]:
        """Return model membership function metadata after fitting."""
        check_is_fitted(self, "model_")
        return self.model_.get_mf_params()

    def _select_model_features(self, x_arr: np.ndarray) -> np.ndarray:
        """Map validated inputs to the fitted model's feature space.

        Identity by default. Estimators that structurally prune features during
        ``fit`` (e.g. DG-TSK) override this to slice inputs to the surviving
        features, so introspection paths match the pruned model width.
        """
        return x_arr

    def rule_activation(self, X: npt.ArrayLike) -> np.ndarray:
        """Return normalized rule activations for the provided inputs."""
        check_is_fitted(self, "model_")
        x_arr = self._select_model_features(validate_data(self, X, reset=False))

        was_training = self.model_.training
        try:
            set_training_flag(self.model_, False)
            with torch.no_grad():
                norm_w = self.model_.forward_antecedents(self._as_tensor_x(x_arr, torch.device(str(self.device))))
        finally:
            set_training_flag(self.model_, was_training)

        return _to_numpy(norm_w)

    def inspect(self) -> dict[str, Any]:
        """Return a structured summary of fitted model state and rule metadata."""
        check_is_fitted(self, "model_")
        return {
            "n_rules": int(self.model_.n_rules),
            "n_inputs": int(self.model_.n_inputs),
            "feature_names": list(self.model_.input_names),
            "rule_base": self.rule_base_,
            "defuzzifier_type": type(self.model_.defuzzifier).__name__,
            "mf_params": self.get_mf_params(),
            "rule_table": self.model_.get_rule_table(),
        }

    def feature_importance(self) -> np.ndarray | None:
        """Compute a normalized feature importance vector from consequent weights."""
        check_is_fitted(self, "model_")
        weights = self.model_.get_consequent_weights()
        if weights is None:
            return None

        consequent_layer = self.model_.consequent_layer
        if hasattr(consequent_layer, "rule_feature_mask"):
            rule_feature_mask = cast(Tensor, consequent_layer.rule_feature_mask)
            weights = weights * rule_feature_mask.unsqueeze(1) if weights.ndim == 3 else weights * rule_feature_mask

        abs_weights = weights.abs()
        if abs_weights.ndim == 3:
            importance = abs_weights.mean(dim=(0, 1))
        elif abs_weights.ndim == 2:
            importance = abs_weights.mean(dim=0)
        else:
            raise ValueError("unsupported consequent weight shape for feature importance")

        return _normalize_importance(importance)


class _BaseClassifierEstimator(ClassifierMixin, _BaseTSKEstimator):  # type: ignore[misc]
    """Abstract base class for all highFIS TSK classifier estimators.

    Implements the full scikit-learn estimator protocol — ``fit``,
    ``predict_proba``, ``predict``, ``score``, ``save`` and ``load`` — and
    delegates only model construction to concrete subclasses via the abstract
    method :meth:`_build_model`.

    Subclasses should implement :meth:`_build_model` and may override
    :meth:`_get_trainer` to supply a custom training strategy.
    Use :meth:`_pre_train_hook` for any per-estimator setup that must run
    just before training starts.  Direct ``fit`` overrides should be avoided.

    Attributes:
        model_: Fitted :class:`~highfis.models.BaseTSK` instance. Available
            after :meth:`fit`.
        classes_: Unique class labels discovered during :meth:`fit`.
        n_features_in_: Number of input features seen during :meth:`fit`.
        feature_names_in_: Array of feature name strings.
        history_: Training history dictionary returned by the underlying model.
        rule_base_: Rule-base type actually used during the last :meth:`fit`.
    """

    @abstractmethod
    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create the concrete TSK classification model."""

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        x_val: npt.ArrayLike | None = None,
        y_val: npt.ArrayLike | None = None,
        metrics: list[str] | None = None,
    ) -> Self:
        """Train the TSK classifier on labeled samples.

        Validation data should be supplied using ``x_val`` and ``y_val``
        when available.
        """
        x_arr, y_arr = validate_data(self, x, y, reset=True)
        n_samples = x_arr.shape[0]
        n_mfs_val = getattr(self, "n_mfs", 3)
        n_mfs_val = 3 if n_mfs_val is None else n_mfs_val
        mf_init_val = getattr(self, "mf_init", "kmeans")
        required_samples = 2 if mf_init_val == "grid" else max(2, n_mfs_val)
        if n_samples < required_samples:
            raise ValueError(
                f"Found array with {n_samples} sample(s). Estimator requires at least {required_samples} samples."
            )
        target_type = type_of_target(y_arr)
        if target_type in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {target_type}")

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        le = LabelEncoder()
        y_idx = le.fit_transform(np.asarray(y_arr))

        input_mfs, _, effective_rule_base = self._build_input_mfs(x_arr)

        self.n_features_in_ = x_arr.shape[1]
        self.classes_ = le.classes_
        self._label_encoder_ = le

        _device = torch.device(str(self.device))
        self.model_ = self._build_model(input_mfs, len(self.classes_), effective_rule_base).to(_device)

        y_t = torch.as_tensor(y_idx, dtype=torch.long, device=_device)

        # Prepare validation tensors if provided via fit.
        x_val_t: torch.Tensor | None = None
        y_val_t: torch.Tensor | None = None
        if (x_val is None) != (y_val is None):
            raise ValueError("x_val and y_val must be provided together")
        if x_val is not None and y_val is not None:
            x_v_arr, y_v_arr = validate_data(self, x_val, y_val, reset=False)
            x_val_t = self._as_tensor_x(x_v_arr, _device)
            y_val_idx = le.transform(np.asarray(y_v_arr))
            y_val_t = torch.as_tensor(y_val_idx, dtype=torch.long, device=_device)

        x_t = self._as_tensor_x(x_arr, _device)
        self.rule_base_ = effective_rule_base
        self._pre_train_hook(self.model_, x_t, y_t)
        _trainer = self.trainer if self.trainer is not None else self._get_trainer()
        self.history_ = _trainer.fit(self.model_, x_t, y_t, x_val=x_val_t, y_val=y_val_t, metrics=metrics)
        return self

    def save(self, path: str) -> None:
        """Persist estimator configuration, model weights and fitted metadata."""
        _fnames: np.ndarray | None = getattr(self, "feature_names_in_", None)
        rules = None
        if hasattr(self.model_, "rule_layer") and hasattr(self.model_.rule_layer, "rules"):
            rules = self.model_.rule_layer.rules
        consequent_mode = getattr(self.model_.consequent_layer, "mode", None)
        checkpoint = self._build_checkpoint_base(
            model_init={
                "input_mfs_config": serialize_input_mfs(self.model_.input_mfs),
                "n_classes": len(self.classes_),
                "rule_base": self.rule_base_,
                "rules": rules,
                "consequent_mode": consequent_mode,
                "is_first_order": self._model_is_first_order(self.model_),
            },
            fitted_attrs={
                "n_features_in": int(self.n_features_in_),
                "feature_names_in": _fnames.tolist() if _fnames is not None else None,
                "classes": self.classes_.tolist(),
                "classes_dtype": str(self.classes_.dtype),
            },
        )
        save_checkpoint(path, checkpoint)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load a persisted estimator created by save."""
        checkpoint = load_checkpoint(path)
        validate_checkpoint_payload(checkpoint, expected_estimator_class=cls.__name__)

        params: dict[str, Any] = dict(checkpoint["estimator_params"])
        if params.get("input_configs") is not None:
            params["input_configs"] = [InputConfig(**c) for c in params["input_configs"]]
        estimator = cls(**params)
        model_init = checkpoint["model_init"]
        estimator.rule_base_ = model_init["rule_base"]

        import inspect

        sig = inspect.signature(estimator._build_model)
        if "rules" in sig.parameters:
            estimator.model_ = estimator._build_model(
                deserialize_input_mfs(model_init["input_mfs_config"]),
                int(model_init["n_classes"]),
                str(model_init["rule_base"]),
                rules=model_init.get("rules"),
            )
        else:
            estimator.model_ = estimator._build_model(
                deserialize_input_mfs(model_init["input_mfs_config"]),
                int(model_init["n_classes"]),
                str(model_init["rule_base"]),
            )

        estimator._restore_consequent_structure(estimator.model_, model_init)

        estimator.model_.load_state_dict(checkpoint["model_state_dict"])
        estimator.model_.to(torch.device(str(estimator.device)))

        fitted = checkpoint["fitted_attrs"]
        estimator.n_features_in_ = int(fitted["n_features_in"])
        if fitted.get("feature_names_in") is not None:
            estimator.feature_names_in_ = np.asarray(fitted["feature_names_in"], dtype=object)
        elif hasattr(estimator, "feature_names_in_"):
            delattr(estimator, "feature_names_in_")
        # Restore ``classes_`` with its original dtype so the reloaded estimator
        # matches the fitted one. Forcing ``object`` breaks scikit-learn metrics
        # (``score``/``cross_val_score`` raise on an "unknown" target type). Older
        # checkpoints lack ``classes_dtype``; fall back to natural inference
        # (which yields int/str rather than object) for those.
        classes_dtype = fitted.get("classes_dtype")
        estimator.classes_ = np.asarray(fitted["classes"], dtype=np.dtype(classes_dtype) if classes_dtype else None)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = estimator.classes_
        estimator._label_encoder_ = label_encoder
        history = checkpoint.get("history")
        estimator.history_ = history if history is not None else {}
        return estimator

    def predict_proba(self, x: npt.ArrayLike) -> np.ndarray:
        """Predict class probabilities for input samples."""
        check_is_fitted(self, "model_")
        x_arr = validate_data(self, x, reset=False)
        device_str = str(self.device).lower()
        x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=torch.device(device_str))
        probs = cast(Any, self.model_).predict_proba(x_tensor)
        return probs.detach().cpu().numpy()

    def predict(self, x: npt.ArrayLike) -> np.ndarray:
        """Predict class labels for input samples."""
        proba = self.predict_proba(x)
        y_idx = np.argmax(proba, axis=1)
        return np.asarray(self._label_encoder_.inverse_transform(y_idx))

    def score(self, X: npt.ArrayLike, y: npt.ArrayLike, sample_weight: npt.ArrayLike | None = None) -> float:
        """Return classification accuracy on the provided dataset."""
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))

    def evaluate(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        metrics: list[str] | None = None,
        sample_weight: npt.ArrayLike | None = None,
    ) -> dict[str, Any]:
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


class _BaseRegressorEstimator(RegressorMixin, _BaseTSKEstimator):  # type: ignore[misc]
    """Abstract base class for all highFIS TSK regressor estimators.

    Implements the full scikit-learn estimator protocol — ``fit``,
    ``predict``, ``score``, ``save`` and ``load`` — and delegates only model
    construction to concrete subclasses via the abstract method
    :meth:`_build_model`.

    Subclasses should implement :meth:`_build_regressor_model` and may
    override :meth:`_get_trainer` to supply a custom training strategy.
    Use :meth:`_pre_train_hook` for any per-estimator setup that must run
    just before training starts.  Direct ``fit`` overrides should be avoided.

    Attributes:
        model_: Fitted :class:`~highfis.models.BaseTSK` instance. Available
            after :meth:`fit`.
        n_features_in_: Number of input features seen during :meth:`fit`.
        feature_names_in_: Array of feature name strings.
        history_: Training history dictionary returned by the underlying model.
        rule_base_: Rule-base type actually used during the last :meth:`fit`.
    """

    @abstractmethod
    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create the concrete TSK regression model."""

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        x_val: npt.ArrayLike | None = None,
        y_val: npt.ArrayLike | None = None,
        metrics: list[str] | None = None,
    ) -> Self:
        """Train the TSK regressor on labeled samples.

        Validation data should be supplied using ``x_val`` and ``y_val``
        when available.
        """
        x_arr, y_arr = validate_data(self, x, y, reset=True)
        n_samples = x_arr.shape[0]
        n_mfs_val = getattr(self, "n_mfs", 3)
        n_mfs_val = 3 if n_mfs_val is None else n_mfs_val
        mf_init_val = getattr(self, "mf_init", "kmeans")
        required_samples = 2 if mf_init_val == "grid" else max(2, n_mfs_val)
        if n_samples < required_samples:
            raise ValueError(
                f"Found array with {n_samples} sample(s). Estimator requires at least {required_samples} samples."
            )

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        input_mfs, _, effective_rule_base = self._build_input_mfs(x_arr)

        self.n_features_in_ = x_arr.shape[1]

        _device = torch.device(str(self.device))
        self.model_ = self._build_regressor_model(input_mfs, effective_rule_base).to(_device)

        y_t = torch.as_tensor(np.asarray(y_arr, dtype=np.float32), dtype=torch.float32, device=_device)

        # Prepare validation tensors if provided via fit.
        x_val_t: torch.Tensor | None = None
        y_val_t: torch.Tensor | None = None
        if (x_val is None) != (y_val is None):
            raise ValueError("x_val and y_val must be provided together")
        if x_val is not None and y_val is not None:
            x_v_arr, y_v_arr = validate_data(self, x_val, y_val, reset=False)
            x_val_t = self._as_tensor_x(x_v_arr, _device)
            y_val_t = torch.as_tensor(np.asarray(y_v_arr, dtype=np.float32), dtype=torch.float32, device=_device)

        x_t = self._as_tensor_x(x_arr, _device)
        self.rule_base_ = effective_rule_base
        self._pre_train_hook(self.model_, x_t, y_t)
        _trainer = self.trainer if self.trainer is not None else self._get_trainer()
        self.history_ = _trainer.fit(self.model_, x_t, y_t, x_val=x_val_t, y_val=y_val_t, metrics=metrics)
        return self

    def save(self, path: str) -> None:
        """Persist estimator configuration, model weights and fitted metadata."""
        _fnames: np.ndarray | None = getattr(self, "feature_names_in_", None)
        rules = None
        if hasattr(self.model_, "rule_layer") and hasattr(self.model_.rule_layer, "rules"):
            rules = self.model_.rule_layer.rules
        consequent_mode = getattr(self.model_.consequent_layer, "mode", None)
        checkpoint = self._build_checkpoint_base(
            model_init={
                "input_mfs_config": serialize_input_mfs(self.model_.input_mfs),
                "rule_base": self.rule_base_,
                "rules": rules,
                "consequent_mode": consequent_mode,
                "is_first_order": self._model_is_first_order(self.model_),
            },
            fitted_attrs={
                "n_features_in": int(self.n_features_in_),
                "feature_names_in": _fnames.tolist() if _fnames is not None else None,
            },
        )
        save_checkpoint(path, checkpoint)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load a persisted estimator created by save."""
        checkpoint = load_checkpoint(path)
        validate_checkpoint_payload(checkpoint, expected_estimator_class=cls.__name__)

        params: dict[str, Any] = dict(checkpoint["estimator_params"])
        if params.get("input_configs") is not None:
            params["input_configs"] = [InputConfig(**c) for c in params["input_configs"]]
        estimator = cls(**params)
        model_init = checkpoint["model_init"]
        estimator.rule_base_ = model_init["rule_base"]

        import inspect

        sig = inspect.signature(estimator._build_regressor_model)
        if "rules" in sig.parameters:
            estimator.model_ = estimator._build_regressor_model(
                deserialize_input_mfs(model_init["input_mfs_config"]),
                str(model_init["rule_base"]),
                rules=model_init.get("rules"),
            )
        else:
            estimator.model_ = estimator._build_regressor_model(
                deserialize_input_mfs(model_init["input_mfs_config"]),
                str(model_init["rule_base"]),
            )

        estimator._restore_consequent_structure(estimator.model_, model_init)

        estimator.model_.load_state_dict(checkpoint["model_state_dict"])
        estimator.model_.to(torch.device(str(estimator.device)))

        fitted = checkpoint["fitted_attrs"]
        estimator.n_features_in_ = int(fitted["n_features_in"])
        if fitted.get("feature_names_in") is not None:
            estimator.feature_names_in_ = np.asarray(fitted["feature_names_in"], dtype=object)
        elif hasattr(estimator, "feature_names_in_"):
            delattr(estimator, "feature_names_in_")
        history = checkpoint.get("history")
        estimator.history_ = history if history is not None else {}
        return estimator

    def predict(self, x: npt.ArrayLike) -> np.ndarray:
        """Predict continuous target values for input samples."""
        check_is_fitted(self, "model_")
        x_arr = validate_data(self, x, reset=False)
        device_str = str(self.device).lower()
        x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=torch.device(device_str))
        preds = cast(Any, self.model_).predict(x_tensor)
        return preds.detach().cpu().numpy()

    def evaluate(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        metrics: list[str] | None = None,
        sample_weight: npt.ArrayLike | None = None,
    ) -> dict[str, Any]:
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
