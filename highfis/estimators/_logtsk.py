from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseTSK
from ..memberships import (
    MembershipFunction,
)
from ..models import (
    LogTSKClassifier,
    LogTSKRegressor,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)


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

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
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
        """Create LogTSKRegressor."""
        return LogTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )
