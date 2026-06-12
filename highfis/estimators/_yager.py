"""Sklearn-compatible estimators for AYA-TSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ..memberships import (
    CompositeExponentialMF,
    MembershipFunction,
)
from ..models import (
    AYATSKClassifierModel,
    AYATSKRegressorModel,
    BaseTSK,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)


def _build_ayatsk_default_input_mfs(
    feature_names: Sequence[str],
    *,
    k: float,
) -> dict[str, list[CompositeExponentialMF]]:
    """Build the paper-style AYATSK default CEMFs.

    The paper initializes three fuzzy sets per feature with centers
    0.0, 0.5, and 1.0, spread 1.0, and a positive lower bound.
    """
    centers = [0.0, 0.5, 1.0]
    return {
        name: [CompositeExponentialMF(center=center, sigma=1.0, k=k) for center in centers] for name in feature_names
    }


def _resolve_ayatsk_default_batch_size(n_samples: int) -> int | None:
    """Resolve the paper-style AYATSK batch size for a dataset."""
    if n_samples < 500:
        return None
    return max(1, round(0.1 * float(n_samples)))


class AYATSKClassifier(_BaseClassifierEstimator):
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
        from highfis import AYATSKClassifier

        clf = AYATSKClassifier(n_mfs=30, random_state=0)
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 3,
        mf_init: str = "grid",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        verbose: bool | int = False,
        rule_base: str | None = "coco",
        batch_size: int | None = None,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 0.0,
        k: float = 10.0,
        device: str = "cpu",
    ) -> None:
        """Initialise an AYATSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init`` falls back to clustering.
            n_mfs: Number of fuzzy sets per feature (default ``3``).
            mf_init: Membership-function initialization strategy.  Defaults
                to ``"grid"`` for the paper-style CEMF initialization.
            sigma_scale: Sigma scale factor for non-default initialisation.
            random_state: Seed for clustering and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.001``).
            verbose: Print per-epoch progress.
            rule_base: Rule-base strategy. Defaults to ``"coco"``.
            batch_size: Mini-batch size. ``None`` applies the paper policy:
                full-batch when ``N < 500`` and ``0.1 * N`` otherwise.
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by
                AYATSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            k: CEMF lower-bound control parameter. Must be ``> 1``.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
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
            weight_decay=weight_decay,
            device=device,
        )
        self.k = k

    def _build_input_mfs(
        self,
        x_arr: np.ndarray,
    ) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        if x_arr.shape[1] < 2:
            raise ValueError(f"AYATSKClassifier requires at least 2 features; got n_features={x_arr.shape[1]}.")
        if (
            self.input_configs is None
            and isinstance(self.mf_init, str)
            and self.mf_init.lower() == "grid"
            and int(self.n_mfs) == 3
        ):
            input_configs = self._resolve_input_configs(x_arr)
            feature_names = [cfg.name for cfg in input_configs]
            input_mfs = _build_ayatsk_default_input_mfs(feature_names, k=self.k)
            return input_mfs, feature_names, self.rule_base if self.rule_base is not None else "coco"
        return super()._build_input_mfs(x_arr)

    def _resolve_default_batch_size(self, n_samples: int) -> int | None:
        return _resolve_ayatsk_default_batch_size(n_samples)

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        x_val: npt.ArrayLike | None = None,
        y_val: npt.ArrayLike | None = None,
        metrics: list[str] | None = None,
    ) -> AYATSKClassifier:
        if float(self.k) <= 1.0:
            raise ValueError("k must be > 1.0")
        original_batch_size = self.batch_size
        try:
            y_arr = np.asarray(y)
            n_samples = y_arr.shape[0] if y_arr.ndim >= 1 else 0
            if original_batch_size is None:
                self.batch_size = self._resolve_default_batch_size(n_samples)
            return super().fit(x, y, x_val=x_val, y_val=y_val, metrics=metrics)
        finally:
            self.batch_size = original_batch_size

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create AYATSKClassifierModel."""
        return AYATSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            rules=rules,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class AYATSKRegressor(_BaseRegressorEstimator):
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
        from highfis import AYATSKRegressor

        reg = AYATSKRegressor(n_mfs=30, random_state=0)
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 3,
        mf_init: str = "grid",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        verbose: bool | int = False,
        rule_base: str | None = "coco",
        batch_size: int | None = None,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 0.0,
        k: float = 10.0,
        device: str = "cpu",
    ) -> None:
        """Initialise an AYATSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init`` falls back to clustering.
            n_mfs: Number of fuzzy sets per feature (default ``3``).
            mf_init: Membership-function initialization strategy.  Defaults
                to ``"grid"`` for the paper-style CEMF initialization.
            sigma_scale: Sigma scale factor for non-default initialisation.
            random_state: Seed for clustering and weight initialisation.
            epochs: Maximum training epochs (default ``200``).
            learning_rate: Adam learning rate (default ``0.001``).
            verbose: Print per-epoch progress.
            rule_base: Rule-base strategy. Defaults to ``"coco"``.
            batch_size: Mini-batch size. ``None`` applies the paper policy:
                full-batch when ``N < 500`` and ``0.1 * N`` otherwise.
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            k: CEMF lower-bound control parameter. Must be ``> 1``.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
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
            weight_decay=weight_decay,
            device=device,
        )
        self.k = k

    def _build_input_mfs(
        self,
        x_arr: np.ndarray,
    ) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        if x_arr.shape[1] < 2:
            raise ValueError(f"AYATSKRegressor requires at least 2 features; got n_features={x_arr.shape[1]}.")
        if (
            self.input_configs is None
            and isinstance(self.mf_init, str)
            and self.mf_init.lower() == "grid"
            and int(self.n_mfs) == 3
        ):
            input_configs = self._resolve_input_configs(x_arr)
            feature_names = [cfg.name for cfg in input_configs]
            input_mfs = _build_ayatsk_default_input_mfs(feature_names, k=self.k)
            return input_mfs, feature_names, self.rule_base if self.rule_base is not None else "coco"
        return super()._build_input_mfs(x_arr)

    def _resolve_default_batch_size(self, n_samples: int) -> int | None:
        return _resolve_ayatsk_default_batch_size(n_samples)

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        x_val: npt.ArrayLike | None = None,
        y_val: npt.ArrayLike | None = None,
        metrics: list[str] | None = None,
    ) -> AYATSKRegressor:
        if float(self.k) <= 1.0:
            raise ValueError("k must be > 1.0")
        original_batch_size = self.batch_size
        try:
            y_arr = np.asarray(y)
            n_samples = y_arr.shape[0] if y_arr.ndim >= 1 else 0
            if original_batch_size is None:
                self.batch_size = self._resolve_default_batch_size(n_samples)
            return super().fit(x, y, x_val=x_val, y_val=y_val, metrics=metrics)
        finally:
            self.batch_size = original_batch_size

    def __sklearn_tags__(self) -> Any:
        """Mark as poor_score: AYATSK is designed for high-dimensional data."""
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create AYATSKRegressorModel."""
        return AYATSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            rules=rules,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )
