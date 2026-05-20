"""Sklearn-compatible estimators for AYA-TSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseTSK
from ..memberships import (
    MembershipFunction,
)
from ..models import (
    AYATSKClassifier,
    AYATSKRegressor,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
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
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
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
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create AYATSKRegressor."""
        return AYATSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )
