import math
from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import torch
from sklearn.utils.validation import check_X_y
from torch import Tensor

from ..clustering import FuzzyCMeans
from ..memberships import (
    ConstantMF,
    GaussianMF,
    MembershipFunction,
)
from ..models import (
    BaseTSK,
    MHTSKClassifierModel,
    MHTSKRegressorModel,
)
from ._base import (
    BatchSizeSpec,
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)


def _fit_fuzzy_c_means_on_head(
    x: np.ndarray,
    subset: np.ndarray,
    instance_sample_fraction: float,
    sample_size: int,
    n_samples: int,
    n_clusters: int,
    fcm_m: float,
    random_state: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if instance_sample_fraction < 1.0 and sample_size < n_samples:
        row_indices = rng.choice(n_samples, size=sample_size, replace=False)
        x_sub = x[row_indices][:, subset]
    else:
        x_sub = x[:, subset]

    model = FuzzyCMeans(n_clusters=n_clusters, m=fcm_m, random_state=random_state)
    model.fit(x_sub)
    if model.cluster_centers_ is None:
        raise RuntimeError("FuzzyCMeans did not converge to a valid solution")
    return model.cluster_centers_.cpu().numpy()


def _populate_head_mfs_and_rules(
    centers: np.ndarray,
    subset: np.ndarray,
    n_clusters: int,
    n_features: int,
    rule_sigma: float,
    feature_names: list[str],
    input_mfs: dict[str, list[MembershipFunction]],
    rules: list[tuple[int, ...]],
) -> None:
    feature_indices: dict[int, list[int]] = {}
    for j, feature_idx in enumerate(subset):
        mf_list = input_mfs[feature_names[feature_idx]]
        start_index = len(mf_list)
        for k in range(n_clusters):
            mf_list.append(GaussianMF(mean=float(centers[k, j]), sigma=rule_sigma))
        feature_indices[feature_idx] = list(range(start_index, start_index + n_clusters))

    for k in range(n_clusters):
        rule_indices: list[int] = []
        for feature_idx in range(n_features):
            if feature_idx in feature_indices:
                rule_indices.append(feature_indices[feature_idx][k])
            else:
                rule_indices.append(0)
        rules.append(tuple(rule_indices))


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
        centers = _fit_fuzzy_c_means_on_head(
            x, subset, instance_sample_fraction, sample_size, n_samples, n_clusters, fcm_m, random_state, rng
        )
        _populate_head_mfs_and_rules(
            centers, subset, n_clusters, n_features, rule_sigma, feature_names, input_mfs, rules
        )

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


def _validate_mhtsk_scale_inputs(
    n_features: int,
    head_size: int | None,
    head_size_ratio: float | None,
    n_heads: int | None,
    fcr_target: float | None,
    h_value: float | None,
    sigma: float,
    xi: float,
) -> None:
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
    _validate_mhtsk_scale_inputs(n_features, head_size, head_size_ratio, n_heads, fcr_target, h_value, sigma, xi)

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
    selected_us = _select_rule_indices(unsupervised_scores, crcr_us)

    selected_s: list[int] = []
    if y is not None and y.ndim == 1 and len(torch.unique(y)) > 1 and crcr_s > 0.0:
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
        selected_s = _select_rule_indices(supervised_scores, crcr_s)

    selected_indices = sorted(set(selected_us) | set(selected_s))
    if len(selected_indices) == 0 and n_rules > 0:
        selected_indices = [int(torch.argmax(unsupervised_scores).item())]
    return selected_indices


def _extract_mhtsk_rule_indices_unsupervised(norm_w: Tensor, crcr_us: float) -> list[int]:
    return _select_rule_indices(torch.max(norm_w, dim=0).values, crcr_us)


def _mhtsk_default_batch_size(n_samples: int) -> int:
    """MHTSK_2025 does not specify a batch size; use 64, matching HDFIS's small-data choice."""
    return 64


class MHTSKClassifier(_BaseClassifierEstimator):
    """Estimator for the multihead Takagi-Sugeno-Kang fuzzy system.

    This estimator supports paper-derived automatic scale parameter resolution
    for head size and number of heads, and it optionally subsamples instances
    when building each head as described in the MHTSK paper.

    MHTSK builds multiple sparse subantecedents from random feature
    subsets and jointly optimizes their rule consequents.

    Reference:
        Z. Bian, Q. Chang, J. Wang and N. R. Pal, "Multihead
        Takagi-Sugeno-Kang Fuzzy System," in IEEE Transactions
        on Fuzzy Systems, vol. 33, no. 8, pp. 2561-2573, Aug. 2025,
        doi: 10.1109/TFUZZ.2025.3569227.

    Example:
        ```python
        from highfis import MHTSKClassifier

        clf = MHTSKClassifier()
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 3,
        n_heads: int | None = None,
        head_size: int | None = None,
        head_size_ratio: float | None = None,
        fcm_m: float = 2.0,
        rule_sigma: float = 1.0,
        fcr_target: float | None = None,
        h_value: float | None = None,
        xi: float = 743.0,
        instance_sample_fraction: float = 0.8,
        rule_extraction: bool = False,
        crcr_us: float = 0.5,
        crcr_s: float = 0.5,
        retrain_after_extraction: bool = True,
        random_state: int | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        batch_size: BatchSizeSpec = "auto",
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        device: str = "cpu",
        eval_metrics_every: int = 1,
        scheduler_class: type[Any] | None = None,
        scheduler_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a MHTSK classifier estimator.

        Args:
            input_configs: Optional list of per-feature input configurations.
                Only the feature names are used for FCM-based MHTSK head construction.
            n_mfs: Number of FCM clusters per head (number of rules generated by each head).
            n_heads: Number of random heads. If ``None``, this is resolved from
                ``head_size`` together with ``fcr_target`` or ``h_value``.
            head_size: Number of features sampled per head. If ``None``, defaults to
                ``round(D * 0.02)`` for ``D <= 5000`` and ``round(D * 0.01)`` for larger inputs.
            head_size_ratio: Alternative relative head size, as a fraction of the input dimension.
            fcm_m: Fuzzification exponent for FCM cluster fitting.
            rule_sigma: Gaussian sigma applied to rule antecedent membership functions.
            fcr_target: Target feature coverage rate for randomly sampled heads.
            h_value: Paper-derived scale constant ``H`` used to compute the number of heads.
                When set, this overrides ``fcr_target``.
            xi: Numeric underflow threshold constant used to bound ``head_size``.
            instance_sample_fraction: Fraction of training instances sampled per head for FCM.
            rule_extraction: If ``True``, perform post-fit rule extraction (MHTSK_RE).
            crcr_us: Unsupervised cumulative rule contribution rate target used in extraction.
            crcr_s: Supervised cumulative rule contribution rate target used in extraction.
            retrain_after_extraction: If ``True``, retrain the extracted rule base after extraction.
            random_state: Random seed for reproducible head construction and FCM initialization.
            epochs: Maximum number of training epochs.
            learning_rate: Adam optimizer learning rate.
            verbose: Verbosity level during training.
            batch_size: Mini-batch size for gradient descent.
            shuffle: Whether to shuffle training samples each epoch.
            ur_weight: Weight of the uncertainty regularization term.
            ur_target: Target firing-level for uncertainty regularization.
            consequent_batch_norm: Apply batch normalization to the consequent layer inputs.
            patience: Early-stopping patience for validation.
            restore_best: Whether to restore the best validation model weights after training.
            weight_decay: Weight decay coefficient for the optimizer.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            eval_metrics_every: Evaluate training metrics every ``n`` epochs; ``0``
                skips them. Each evaluation is an extra forward pass over the training
                set and only fills ``history_["train_<metric>"]``; early stopping uses
                validation metrics regardless.
            scheduler_class: Learning-rate scheduler *class* (e.g.
                ``torch.optim.lr_scheduler.StepLR``), not an instance -- the optimiser
                it must bind to is only built inside ``fit``.
            scheduler_params: Keyword arguments for ``scheduler_class``.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init="fcm",
            sigma_scale=1.0,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base="custom",
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=None,
            patience=patience,
            restore_best=restore_best,
            weight_decay=weight_decay,
            device=device,
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
        )
        self.n_heads = n_heads
        self.head_size = head_size
        self.head_size_ratio = head_size_ratio
        self.fcm_m = fcm_m
        self.rule_sigma = rule_sigma
        self.fcr_target = fcr_target
        self.h_value = h_value
        self.xi = xi
        self.instance_sample_fraction = instance_sample_fraction
        self.rule_extraction = rule_extraction
        self.crcr_us = crcr_us
        self.crcr_s = crcr_s
        self.retrain_after_extraction = retrain_after_extraction

    def _paper_batch_size(self, n_samples: int) -> int | None:
        """MHTSK_2025 specifies no batch size; default to 64."""
        return _mhtsk_default_batch_size(n_samples)

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:  # type: ignore[override]
        feature_names = self._resolve_feature_names(x_arr)
        head_size, n_heads = _resolve_mhtsk_scale_parameters(
            n_features=x_arr.shape[1],
            head_size=self.head_size,
            head_size_ratio=self.head_size_ratio,
            n_heads=self.n_heads,
            fcr_target=self.fcr_target,
            h_value=self.h_value,
            sigma=self.rule_sigma,
            xi=self.xi,
        )
        input_mfs, rules, rule_feature_mask = _build_mhtsk_input_mfs(
            x_arr,
            feature_names=feature_names,
            n_heads=n_heads,
            head_size=head_size,
            n_clusters=int(self.n_mfs),
            fcm_m=self.fcm_m,
            rule_sigma=self.rule_sigma,
            instance_sample_fraction=self.instance_sample_fraction,
            random_state=self.random_state,
        )
        self._mhtsk_rules = rules
        self._mhtsk_rule_feature_mask = torch.as_tensor(rule_feature_mask, dtype=torch.bool)
        return input_mfs, feature_names, "custom"

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        rules_to_use = rules if rules is not None else self._mhtsk_rules
        return MHTSKClassifierModel(
            input_mfs,
            self._mhtsk_rule_feature_mask,
            rules_to_use,
            n_classes=n_classes,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )

    def _build_extracted_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_indices: list[int],
    ) -> None:
        if not rule_indices:
            raise ValueError("At least one rule must be selected for extraction")
        self._mhtsk_rules = [self._mhtsk_rules[i] for i in rule_indices]
        self._mhtsk_rule_feature_mask = self._mhtsk_rule_feature_mask[rule_indices]
        self.model_ = self._build_model(input_mfs, len(self.classes_), self.rule_base_)

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
        metrics: list[str] | None = None,
    ) -> Self:
        """Train the MHTSK classifier and optionally extract rules.

        After the base training step, if ``rule_extraction`` is enabled, the
        firing-strength matrix is used to select a compact rule subset via the
        CRCR criterion.  When ``retrain_after_extraction`` is also set, a
        second training pass is performed on the reduced model.
        """
        x_arr, y_arr = check_X_y(x, y)
        super().fit(x, y, x_val=x_val, y_val=y_val, metrics=metrics)

        if not bool(self.rule_extraction):
            return self

        _device = torch.device(str(self.device))
        x_t = self._as_tensor_x(x_arr, _device)
        self.model_.eval()
        with torch.no_grad():
            norm_w = self.model_.forward_antecedents(x_t)

        y_t = torch.as_tensor(self._label_encoder_.transform(np.asarray(y_arr)), dtype=torch.long)
        selected = _extract_mhtsk_rule_indices(norm_w, y_t, self.crcr_us, self.crcr_s)
        self._extracted_rule_indices_ = selected

        input_mfs = self.model_.input_mfs
        self._build_extracted_model(input_mfs, selected)

        if self.retrain_after_extraction:
            x_val_t: torch.Tensor | None = None
            y_val_t: torch.Tensor | None = None
            if x_val is not None and y_val is not None:
                x_v_arr, y_v_arr = check_X_y(x_val, y_val)
                x_val_t = self._as_tensor_x(x_v_arr, _device)
                y_val_t = torch.as_tensor(
                    self._label_encoder_.transform(np.asarray(y_v_arr)),
                    dtype=torch.long,
                )
            trainer = self.trainer if self.trainer is not None else self._get_trainer()
            self.history_ = trainer.fit(self.model_, x_t, y_t, x_val=x_val_t, y_val=y_val_t, metrics=metrics)
        return self


class MHTSKRegressor(_BaseRegressorEstimator):
    """Estimator for the multihead Takagi-Sugeno-Kang fuzzy system.

    The regressor uses the same MHTSK head construction and scale parameter
    strategy as the classifier. Rule extraction is currently implemented with
    an unsupervised scheme only, since regression does not provide class labels
    for the Mann-Whitney based selection used in classification.

    MHTSK builds multiple sparse subantecedents from random feature
    subsets and jointly optimizes their rule consequents.

    Reference:
        Z. Bian, Q. Chang, J. Wang and N. R. Pal, "Multihead
        Takagi-Sugeno-Kang Fuzzy System," in IEEE Transactions
        on Fuzzy Systems, vol. 33, no. 8, pp. 2561-2573, Aug. 2025,
        doi: 10.1109/TFUZZ.2025.3569227.

    Example:
        ```python
        from highfis import MHTSKRegressor

        reg = MHTSKRegressor()
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 3,
        n_heads: int | None = None,
        head_size: int | None = None,
        head_size_ratio: float | None = None,
        fcm_m: float = 2.0,
        rule_sigma: float = 1.0,
        fcr_target: float | None = None,
        h_value: float | None = None,
        xi: float = 743.0,
        instance_sample_fraction: float = 0.8,
        rule_extraction: bool = False,
        crcr_us: float = 0.5,
        retrain_after_extraction: bool = True,
        random_state: int | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        batch_size: BatchSizeSpec = "auto",
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        device: str = "cpu",
        eval_metrics_every: int = 1,
        scheduler_class: type[Any] | None = None,
        scheduler_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a MHTSK regressor estimator.

        Args:
            input_configs: Optional list of per-feature input configurations.
                Only the feature names are used for FCM-based MHTSK head construction.
            n_mfs: Number of FCM clusters per head (number of rules generated by each head).
            n_heads: Number of random heads. If ``None``, this is resolved from
                ``head_size`` together with ``fcr_target`` or ``h_value``.
            head_size: Number of features sampled per head. If ``None``, defaults to
                ``round(D * 0.02)`` for ``D <= 5000`` and ``round(D * 0.01)`` for larger inputs.
            head_size_ratio: Alternative relative head size, as a fraction of the input dimension.
            fcm_m: Fuzzification exponent for FCM cluster fitting.
            rule_sigma: Gaussian sigma applied to rule antecedent membership functions.
            fcr_target: Target feature coverage rate for randomly sampled heads.
            h_value: Paper-derived scale constant ``H`` used to compute the number of heads.
                When set, this overrides ``fcr_target``.
            xi: Numeric underflow threshold constant used to bound ``head_size``.
            instance_sample_fraction: Fraction of training instances sampled per head for FCM.
            rule_extraction: If ``True``, perform post-fit rule extraction.
            crcr_us: Unsupervised cumulative rule contribution rate target used in extraction.
            retrain_after_extraction: If ``True``, retrain the extracted rule base after extraction.
            random_state: Random seed for reproducible head construction and FCM initialization.
            epochs: Maximum number of training epochs.
            learning_rate: Adam optimizer learning rate.
            verbose: Verbosity level during training.
            batch_size: Mini-batch size for gradient descent.
            shuffle: Whether to shuffle training samples each epoch.
            ur_weight: Weight of the uncertainty regularization term.
            ur_target: Target firing-level for uncertainty regularization.
            consequent_batch_norm: Apply batch normalization to the consequent layer inputs.
            patience: Early-stopping patience for validation.
            restore_best: Whether to restore the best validation model weights after training.
            weight_decay: Weight decay coefficient for the optimizer.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            eval_metrics_every: Evaluate training metrics every ``n`` epochs; ``0``
                skips them. Each evaluation is an extra forward pass over the training
                set and only fills ``history_["train_<metric>"]``; early stopping uses
                validation metrics regardless.
            scheduler_class: Learning-rate scheduler *class* (e.g.
                ``torch.optim.lr_scheduler.StepLR``), not an instance -- the optimiser
                it must bind to is only built inside ``fit``.
            scheduler_params: Keyword arguments for ``scheduler_class``.

        Notes:
            The regressor supports only unsupervised rule extraction via
            ``crcr_us`` because no label-based Mann-Whitney selection is available.
        """
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init="fcm",
            sigma_scale=1.0,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base="custom",
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=None,
            patience=patience,
            restore_best=restore_best,
            weight_decay=weight_decay,
            device=device,
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
        )
        self.n_heads = n_heads
        self.head_size = head_size
        self.head_size_ratio = head_size_ratio
        self.fcm_m = fcm_m
        self.rule_sigma = rule_sigma
        self.fcr_target = fcr_target
        self.h_value = h_value
        self.xi = xi
        self.instance_sample_fraction = instance_sample_fraction
        self.rule_extraction = rule_extraction
        self.crcr_us = crcr_us
        self.retrain_after_extraction = retrain_after_extraction

    def _paper_batch_size(self, n_samples: int) -> int | None:
        """MHTSK_2025 specifies no batch size; default to 64."""
        return _mhtsk_default_batch_size(n_samples)

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:  # type: ignore[override]
        feature_names = self._resolve_feature_names(x_arr)
        head_size, n_heads = _resolve_mhtsk_scale_parameters(
            n_features=x_arr.shape[1],
            head_size=self.head_size,
            head_size_ratio=self.head_size_ratio,
            n_heads=self.n_heads,
            fcr_target=self.fcr_target,
            h_value=self.h_value,
            sigma=self.rule_sigma,
            xi=self.xi,
        )
        input_mfs, rules, rule_feature_mask = _build_mhtsk_input_mfs(
            x_arr,
            feature_names=feature_names,
            n_heads=n_heads,
            head_size=head_size,
            n_clusters=int(self.n_mfs),
            fcm_m=self.fcm_m,
            rule_sigma=self.rule_sigma,
            instance_sample_fraction=self.instance_sample_fraction,
            random_state=self.random_state,
        )
        self._mhtsk_rules = rules
        self._mhtsk_rule_feature_mask = torch.as_tensor(rule_feature_mask, dtype=torch.bool)
        return input_mfs, feature_names, "custom"

    def _build_extracted_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_indices: list[int],
    ) -> None:
        if not rule_indices:
            raise ValueError("At least one rule must be selected for extraction")
        self._mhtsk_rules = [self._mhtsk_rules[i] for i in rule_indices]
        self._mhtsk_rule_feature_mask = self._mhtsk_rule_feature_mask[rule_indices]
        self.model_ = self._build_regressor_model(input_mfs, self.rule_base_, None)

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
        metrics: list[str] | None = None,
    ) -> Self:
        """Train the MHTSK regressor and optionally extract rules.

        After the base training step, if ``rule_extraction`` is enabled, rules
        are selected via the unsupervised CRCR criterion on the firing-strength
        matrix.  When ``retrain_after_extraction`` is also set, a second
        training pass is performed on the reduced model.
        """
        x_arr, y_arr = check_X_y(x, y)
        super().fit(x, y, x_val=x_val, y_val=y_val, metrics=metrics)

        if not bool(self.rule_extraction):
            return self

        _device = torch.device(str(self.device))
        x_t = self._as_tensor_x(x_arr, _device)
        self.model_.eval()
        with torch.no_grad():
            norm_w = self.model_.forward_antecedents(x_t)

        selected = _extract_mhtsk_rule_indices_unsupervised(norm_w, self.crcr_us)
        self._extracted_rule_indices_ = selected

        input_mfs = self.model_.input_mfs
        self._build_extracted_model(input_mfs, selected)

        if self.retrain_after_extraction:
            y_t = torch.as_tensor(np.asarray(y_arr), dtype=self._model_dtype())
            x_val_t: torch.Tensor | None = None
            y_val_t: torch.Tensor | None = None
            if x_val is not None and y_val is not None:
                x_v_arr, y_v_arr = check_X_y(x_val, y_val)
                x_val_t = self._as_tensor_x(x_v_arr, _device)
                y_val_t = torch.as_tensor(np.asarray(y_v_arr), dtype=self._model_dtype())
            trainer = self.trainer if self.trainer is not None else self._get_trainer()
            self.history_ = trainer.fit(self.model_, x_t, y_t, x_val=x_val_t, y_val=y_val_t, metrics=metrics)
        return self

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        rules_to_use = rules if rules is not None else self._mhtsk_rules
        return MHTSKRegressorModel(
            input_mfs,
            self._mhtsk_rule_feature_mask,
            rules_to_use,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )
