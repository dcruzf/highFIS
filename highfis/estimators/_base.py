"""Base estimator classes, input configuration, and shared utilities."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self, cast

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch import Tensor

from ..base import BaseTSK
from ..clustering import FuzzyCMeans
from ..clustering import KMeans as TorchKMeans
from ..memberships import (
    ConstantMF,
    DimensionDependentGaussianMF,
    GaussianMF,
    GaussianPiMF,
    MembershipFunction,
)
from ..metrics import compute_metrics
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
    km = TorchKMeans(n_clusters=n_clusters, random_state=random_state)
    km.fit(x)
    if km.cluster_centers_ is None:
        raise RuntimeError("KMeans did not compute cluster centers")

    if hasattr(km.cluster_centers_, "cpu"):
        centers = km.cluster_centers_.cpu().numpy()
    else:
        centers = np.asarray(km.cluster_centers_)

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


def _build_fuzzy_c_means_input_mfs(
    x: np.ndarray,
    n_clusters: int,
    m: float,
    sigma_scale: float,
    feature_names: list[str],
    random_state: int | None,
) -> dict[str, list[GaussianMF]]:
    r"""Build Gaussian MFs via fuzzy C-means cluster initialization.

    The MF means are placed at the FCM centroids. Each sigma is sampled
    from ``N(h, 0.2)`` where ``h`` is the cluster-specific, feature-wise
    spread scaled by ``sigma_scale``.
    """
    model = FuzzyCMeans(
        n_clusters=n_clusters,
        m=m,
        random_state=random_state,
    )
    model.fit(x)
    if model.cluster_centers_ is None or model.membership_ is None:
        raise RuntimeError("FuzzyCMeans did not converge to a valid solution")

    centers: np.ndarray = model.cluster_centers_.cpu().numpy()
    membership: np.ndarray = model.membership_.cpu().numpy()
    input_mfs: dict[str, list[GaussianMF]] = {}
    rng = np.random.default_rng(random_state)

    for d, name in enumerate(feature_names):
        col = x[:, d]
        center_col = centers[:, d]
        mfs: list[GaussianMF] = []
        for r in range(n_clusters):
            c = float(center_col[r])
            weights = membership[:, r] ** float(model.m)
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


def _build_mhtsk_input_mfs(
    x: np.ndarray,
    feature_names: list[str],
    n_heads: int,
    head_size: int,
    n_clusters: int,
    fcm_m: float,
    rule_sigma: float,
    instance_sample_fraction: float,
    random_state: int | None,
) -> tuple[dict[str, list[MembershipFunction]], list[tuple[int, ...]], np.ndarray]:
    """Build sparse partial-rule MFs and rule indices for MHTSK.

    Each feature receives a constant "don't care" MF plus all cluster MFs
    produced by FCM for any head where the feature is active.

    Each head may also subsample data instances to match the paper's
    random head construction process.
    """
    n_samples, n_features = x.shape
    if head_size <= 0 or head_size > n_features:
        raise ValueError("head_size must be between 1 and the number of features")
    if n_heads <= 0:
        raise ValueError("n_heads must be > 0")
    if not 0.0 < instance_sample_fraction <= 1.0:
        raise ValueError("instance_sample_fraction must be in (0, 1]")

    rng = np.random.default_rng(random_state)
    input_mfs: dict[str, list[MembershipFunction]] = {name: [ConstantMF(1.0)] for name in feature_names}
    rules: list[tuple[int, ...]] = []

    sample_size = max(1, round(n_samples * instance_sample_fraction))

    for _ in range(n_heads):
        subset = rng.choice(n_features, size=head_size, replace=False)
        if instance_sample_fraction < 1.0 and sample_size < n_samples:
            row_indices = rng.choice(n_samples, size=sample_size, replace=False)
            x_sub = x[row_indices][:, subset]
        else:
            x_sub = x[:, subset]

        model = FuzzyCMeans(n_clusters=n_clusters, m=fcm_m, random_state=random_state)
        model.fit(x_sub)
        if model.cluster_centers_ is None:
            raise RuntimeError("FuzzyCMeans did not converge to a valid solution")
        centers = model.cluster_centers_.cpu().numpy()

        feature_indices: dict[int, list[int]] = {}
        for j, feature_idx in enumerate(subset):
            mf_list = input_mfs[feature_names[feature_idx]]
            start_index = len(mf_list)
            for k in range(n_clusters):
                mf_list.append(GaussianMF(mean=float(centers[k, j]), sigma=float(rule_sigma)))
            feature_indices[feature_idx] = list(range(start_index, start_index + n_clusters))

        for k in range(n_clusters):
            rule_indices: list[int] = []
            for feature_idx in range(n_features):
                if feature_idx in feature_indices:
                    rule_indices.append(feature_indices[feature_idx][k])
                else:
                    rule_indices.append(0)
            rules.append(tuple(rule_indices))

    rule_feature_mask = np.zeros((len(rules), n_features), dtype=bool)
    for r, rule in enumerate(rules):
        for j, mf_idx in enumerate(rule):
            rule_feature_mask[r, j] = mf_idx != 0

    return input_mfs, rules, rule_feature_mask


def feature_coverage_rate(n_features: int, head_size: int, n_heads: int) -> float:
    """Compute the feature coverage rate (FCR) for MHTSK heads.

    FCR is the expected proportion of original features that are selected at
    least once across ``n_heads`` random subsets of size ``head_size``.
    """
    if n_features <= 0:
        raise ValueError("n_features must be > 0")
    if head_size <= 0 or head_size > n_features:
        raise ValueError("head_size must be between 1 and n_features")
    if n_heads < 0:
        raise ValueError("n_heads must be >= 0")

    return 1.0 - (1.0 - float(head_size) / float(n_features)) ** float(n_heads)


def _resolve_mhtsk_scale_parameters(
    n_features: int,
    head_size: int | None,
    head_size_ratio: float | None,
    n_heads: int | None,
    fcr_target: float | None,
    h_value: float | None,
    sigma: float,
    xi: float,
) -> tuple[int, int]:
    """Resolve MHTSK scale parameters using paper-derived defaults.

    This helper reproduces the paper's strategy for selecting rule length and
    number of heads while allowing user override.
    """
    if n_features <= 0:
        raise ValueError("n_features must be > 0")
    if head_size is not None and (head_size <= 0 or head_size > n_features):
        raise ValueError("head_size must be between 1 and the number of features")
    if n_heads is not None and n_heads <= 0:
        raise ValueError("n_heads must be > 0")
    if head_size_ratio is not None and not (0.0 < head_size_ratio <= 1.0):
        raise ValueError("head_size_ratio must be in (0, 1]")
    if fcr_target is not None and not (0.0 < fcr_target < 1.0):
        raise ValueError("fcr_target must be in (0, 1)")
    if h_value is not None and h_value <= 0.0:
        raise ValueError("h_value must be > 0")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if xi <= 0.0:
        raise ValueError("xi must be > 0")

    max_head_size = min(n_features, max(1, math.floor(2.0 * xi * sigma * sigma)))

    if head_size is None:
        if head_size_ratio is not None:
            head_size = max(1, min(n_features, round(n_features * head_size_ratio)))
        else:
            head_size = max(1, round(n_features * 0.02)) if n_features <= 5000 else max(1, round(n_features * 0.01))
    head_size = min(head_size, max_head_size, n_features)

    if n_heads is None:
        H = h_value if h_value is not None else -math.log(1.0 - (fcr_target if fcr_target is not None else 0.85))
        n_heads = math.ceil(H * n_features / head_size)

    return head_size, n_heads


def _rankdata(values: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(values), dtype=np.float64)
    order = np.argsort(values)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _normal_survival(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _mann_whitney_p_value(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 1.0

    combined = np.concatenate([left, right]).astype(np.float64)
    ranks = _rankdata(combined)
    n1 = float(left.size)
    n2 = float(right.size)
    ra = float(np.sum(ranks[: left.size]))
    u1 = ra - (n1 * (n1 + 1.0) / 2.0)
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1.0) / 12.0)
    if sigma == 0.0:  # pragma: no cover
        return 1.0

    z = (u - n1 * n2 / 2.0) / sigma
    return min(1.0, 2.0 * _normal_survival(abs(z)))


def _select_rule_indices(scores: Tensor, crcr: float) -> list[int]:
    if not 0.0 <= float(crcr) <= 1.0:
        raise ValueError("crcr must be between 0 and 1")

    n_rules = int(scores.numel())
    if n_rules == 0:
        return []
    if crcr <= 0.0:
        return []
    if crcr >= 1.0:
        return list(range(n_rules))

    sorted_indices = torch.argsort(scores, descending=True)
    sorted_scores = scores[sorted_indices]

    total = float(torch.sum(sorted_scores).item())
    if total <= 0.0:
        count = max(1, math.ceil(float(crcr) * n_rules))
        return sorted_indices[:count].tolist()

    rcr = sorted_scores / total
    cumulative = torch.cumsum(rcr, dim=0)
    threshold = torch.tensor(float(crcr), dtype=cumulative.dtype)
    count = int(torch.searchsorted(cumulative, threshold, right=False).item() + 1)
    return sorted_indices[: max(1, count)].tolist()


def _extract_mhtsk_rule_indices(
    norm_w: Tensor,
    y: Tensor | None,
    crcr_us: float,
    crcr_s: float,
) -> list[int]:
    n_rules = int(norm_w.shape[1])
    if n_rules == 0:
        return []

    unsupervised_scores = torch.max(norm_w, dim=0).values
    selected_us = _select_rule_indices(unsupervised_scores, float(crcr_us))

    selected_s: list[int] = []
    if y is not None and y.ndim == 1 and len(torch.unique(y)) > 1 and float(crcr_s) > 0.0:
        unique_labels = torch.unique(y)
        groups: list[np.ndarray] = []
        for label in unique_labels:
            mask = y == label
            groups.append(norm_w[mask].cpu().numpy())

        supervised_scores = torch.zeros(n_rules, dtype=torch.float64)
        for r in range(n_rules):
            p_min = 1.0
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    p_val = _mann_whitney_p_value(groups[i][:, r], groups[j][:, r])
                    p_min = min(p_min, p_val)
            supervised_scores[r] = 1.0 - p_min
        selected_s = _select_rule_indices(supervised_scores, float(crcr_s))

    selected_indices = sorted(set(selected_us) | set(selected_s))
    if len(selected_indices) == 0 and n_rules > 0:
        selected_indices = [int(torch.argmax(unsupervised_scores).item())]
    return selected_indices


def _extract_mhtsk_rule_indices_unsupervised(norm_w: Tensor, crcr_us: float) -> list[int]:
    return _select_rule_indices(torch.max(norm_w, dim=0).values, float(crcr_us))


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
                    k=float(k),
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
                Cui et al. (IJCNN 2021). ``"fcm"`` derives MF centres from
                fuzzy C-means cluster centroids and computes sigmas from the
                resulting fuzzy memberships. ``"grid"`` places centres on a
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
                active when ``x_val`` and ``y_val`` are provided.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
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

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        """Build MFs and resolve rule_base from the initialization mode."""
        init = str(self.mf_init).lower()
        if init not in {"kmeans", "fcm", "grid"}:
            raise ValueError(f"mf_init must be 'kmeans', 'fcm' or 'grid', got '{self.mf_init}'")

        if init in {"kmeans", "fcm"}:
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
                if init == "kmeans":
                    input_mfs = _build_kmeans_input_mfs(
                        x_arr,
                        n_clusters=int(self.n_mfs),
                        sigma_scale=effective_sigma_scale,
                        feature_names=feature_names,
                        random_state=self.random_state,
                    )
                else:
                    input_mfs = _build_fuzzy_c_means_input_mfs(
                        x_arr,
                        n_clusters=int(self.n_mfs),
                        m=2.0,
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
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create the concrete TSK classification model."""

    # -- sklearn API ------------------------------------------------------

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
    ) -> Self:
        """Train the TSK classifier on labeled samples.

        Validation data should be supplied using ``x_val`` and ``y_val``
        when available.
        """
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

        # Prepare validation tensors if provided via fit.
        x_val_t: torch.Tensor | None = None
        y_val_t: torch.Tensor | None = None
        if (x_val is None) ^ (y_val is None):
            raise ValueError("x_val and y_val must be provided together")
        if x_val is not None and y_val is not None:
            x_v_arr, y_v_arr = check_X_y(x_val, y_val)
            x_val_t = self._as_tensor_x(x_v_arr)
            y_val_idx = le.transform(np.asarray(y_v_arr))
            y_val_t = torch.as_tensor(y_val_idx, dtype=torch.long)

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
        params = self.get_params(deep=False)
        if params.get("input_configs") is not None:
            params["input_configs"] = [
                {"name": c.name, "n_mfs": c.n_mfs, "overlap": c.overlap, "margin": c.margin}
                for c in params["input_configs"]
            ]
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

    def save(self, path: str) -> None:
        """Persist estimator configuration, model weights and fitted metadata."""
        checkpoint = self._build_checkpoint_base(
            model_init={
                "input_mfs_config": serialize_input_mfs(self.model_.input_mfs),
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

        params: dict[str, Any] = dict(checkpoint["estimator_params"])
        if params.get("input_configs") is not None:
            params["input_configs"] = [InputConfig(**c) for c in params["input_configs"]]
        estimator = cls(**params)
        model_init = checkpoint["model_init"]
        estimator.rule_base_ = model_init["rule_base"]
        estimator.model_ = estimator._build_model(
            deserialize_input_mfs(model_init["input_mfs_config"]),
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

    def get_mf_params(self) -> dict[str, list[dict[str, Any]]]:
        """Return model membership function metadata after fitting."""
        check_is_fitted(self, "model_")
        return self.model_.get_mf_params()

    def rule_activation(self, X: Any) -> np.ndarray:
        """Return normalized rule activations for the provided inputs."""
        check_is_fitted(self, "model_")
        x_arr = check_array(X)
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_arr.shape[1]}")

        was_training = self.model_.training
        try:
            self.model_.eval()
            with torch.no_grad():
                norm_w = self.model_.forward_antecedents(self._as_tensor_x(x_arr))
        finally:
            self.model_.train(was_training)

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
                Cui et al. (IJCNN 2021). ``"fcm"`` derives MF centres from
                fuzzy C-means cluster centroids and computes sigmas from the
                resulting fuzzy memberships. ``"grid"`` places centres on a
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
                active when ``x_val`` and ``y_val`` are provided.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
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

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        """Build MFs and resolve rule_base from the initialization mode."""
        init = str(self.mf_init).lower()
        if init not in {"kmeans", "fcm", "grid"}:
            raise ValueError(f"mf_init must be 'kmeans', 'fcm' or 'grid', got '{self.mf_init}'")

        if init in {"kmeans", "fcm"}:
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
                if init == "kmeans":
                    input_mfs = _build_kmeans_input_mfs(
                        x_arr,
                        n_clusters=int(self.n_mfs),
                        sigma_scale=effective_sigma_scale,
                        feature_names=feature_names,
                        random_state=self.random_state,
                    )
                else:
                    input_mfs = _build_fuzzy_c_means_input_mfs(
                        x_arr,
                        n_clusters=int(self.n_mfs),
                        m=2.0,
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
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create the concrete TSK regression model."""

    # -- sklearn API ------------------------------------------------------

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
    ) -> Self:
        """Train the TSK regressor on labeled samples.

        Validation data should be supplied using ``x_val`` and ``y_val``
        when available.
        """
        x_arr, y_arr = check_X_y(x, y)

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        input_mfs, feature_names, effective_rule_base = self._build_input_mfs(x_arr)

        self.n_features_in_ = x_arr.shape[1]
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)

        self.model_ = self._build_regressor_model(input_mfs, effective_rule_base)

        y_t = torch.as_tensor(np.asarray(y_arr, dtype=np.float32), dtype=torch.float32)

        # Prepare validation tensors if provided via fit.
        x_val_t: torch.Tensor | None = None
        y_val_t: torch.Tensor | None = None
        if (x_val is None) ^ (y_val is None):
            raise ValueError("x_val and y_val must be provided together")
        if x_val is not None and y_val is not None:
            x_v_arr, y_v_arr = check_X_y(x_val, y_val)
            x_val_t = self._as_tensor_x(x_v_arr)
            y_val_t = torch.as_tensor(np.asarray(y_v_arr, dtype=np.float32), dtype=torch.float32)

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
        params = self.get_params(deep=False)
        if params.get("input_configs") is not None:
            params["input_configs"] = [
                {"name": c.name, "n_mfs": c.n_mfs, "overlap": c.overlap, "margin": c.margin}
                for c in params["input_configs"]
            ]
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

    def save(self, path: str) -> None:
        """Persist estimator configuration, model weights and fitted metadata."""
        checkpoint = self._build_checkpoint_base(
            model_init={
                "input_mfs_config": serialize_input_mfs(self.model_.input_mfs),
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

        params: dict[str, Any] = dict(checkpoint["estimator_params"])
        if params.get("input_configs") is not None:
            params["input_configs"] = [InputConfig(**c) for c in params["input_configs"]]
        estimator = cls(**params)
        model_init = checkpoint["model_init"]
        estimator.rule_base_ = model_init["rule_base"]
        estimator.model_ = estimator._build_regressor_model(
            deserialize_input_mfs(model_init["input_mfs_config"]),
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

    def get_mf_params(self) -> dict[str, list[dict[str, Any]]]:
        """Return model membership function metadata after fitting."""
        check_is_fitted(self, "model_")
        return self.model_.get_mf_params()

    def rule_activation(self, X: Any) -> np.ndarray:
        """Return normalized rule activations for the provided inputs."""
        check_is_fitted(self, "model_")
        x_arr = check_array(X)
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_arr.shape[1]}")

        was_training = self.model_.training
        try:
            self.model_.eval()
            with torch.no_grad():
                norm_w = self.model_.forward_antecedents(self._as_tensor_x(x_arr))
        finally:
            self.model_.train(was_training)

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


# =====================================================================
# HTSK Estimators
# =====================================================================
