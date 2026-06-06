"""Sklearn-compatible estimators for Dombi-TSK and ADMTSK models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from ..base import BaseTSK
from ..memberships import (
    GaussianMF,
    MembershipFunction,
)
from ..models import (
    ADMTSKClassifierModel,
    ADMTSKRegressorModel,
    DombiTSKClassifierModel,
    DombiTSKRegressorModel,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
    _wrap_composite_gaussian_input_mfs,
)


def _build_admtsk_default_input_mfs(n_features: int) -> dict[str, list[GaussianMF]]:
    """Build default ADMTSK Gaussian MFs before CGMF wrapping.

    Uses three MFs per feature with centers [0.0, 0.5, 1.0] and sigma=1.0.
    """
    centers = [0.0, 0.5, 1.0]
    sigma = 1.0
    return {f"x{i + 1}": [GaussianMF(mean=center, sigma=sigma) for center in centers] for i in range(n_features)}


class DombiTSKClassifier(_BaseClassifierEstimator):
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
        from highfis import DombiTSKClassifier

        clf = DombiTSKClassifier(n_mfs=30, random_state=0)
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int | None = None,
        mf_init: str | None = None,
        sigma_scale: float | str | None = None,
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
        lambda_: float | None = None,
        lower_bound: float | None = None,
        zero_consequent_init: bool | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialise a DombiTSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
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
            weight_decay: L2 weight decay for consequent parameters.
            lambda_: Dombi parameter ``λ > 0``.
            lower_bound: Lower bound for Composite GMF.
            zero_consequent_init: If ``True`` (default), initialize consequent parameters to zero.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
        """
        resolved_n_mfs = 5 if n_mfs is None else n_mfs
        resolved_mf_init = "kmeans" if mf_init is None else mf_init
        resolved_sigma_scale = 1.0 if sigma_scale is None else sigma_scale
        resolved_rule_base = rule_base
        resolved_lambda = 1.0 if lambda_ is None else float(lambda_)
        resolved_lower_bound = (1.0 / math.e) if lower_bound is None else float(lower_bound)
        resolved_zero_consequent_init = True if zero_consequent_init is None else zero_consequent_init
        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init=resolved_mf_init,
            sigma_scale=resolved_sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=resolved_rule_base,
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
        self.lambda_ = resolved_lambda
        self.lower_bound = resolved_lower_bound
        self.zero_consequent_init = resolved_zero_consequent_init

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        use_paper_default = (
            self.input_configs is None
            and isinstance(self.mf_init, str)
            and self.mf_init.lower() == "grid"
            and int(self.n_mfs) == 3
        )
        if use_paper_default:
            feature_names = [f"x{i + 1}" for i in range(x_arr.shape[1])]
            input_mfs = _build_admtsk_default_input_mfs(x_arr.shape[1])
            effective_rule_base = self.rule_base if self.rule_base is not None else "coco"
            return (
                _wrap_composite_gaussian_input_mfs(input_mfs, eps=self.lower_bound),
                feature_names,
                effective_rule_base,
            )

        input_mfs, feature_names, effective_rule_base = super()._build_input_mfs(x_arr)
        return (
            _wrap_composite_gaussian_input_mfs(input_mfs, eps=self.lower_bound),
            feature_names,
            effective_rule_base,
        )

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create DombiTSKClassifierModel."""
        return DombiTSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            lambda_=self.lambda_,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            zero_consequent_init=self.zero_consequent_init,
        )


class DombiTSKRegressor(_BaseRegressorEstimator):
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
        from highfis import DombiTSKRegressor

        reg = DombiTSKRegressor(n_mfs=30, random_state=0)
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
        epochs: int = 100,
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
        device: str = "cpu",
    ) -> None:
        """Initialise a DombiTSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
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

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create DombiTSKRegressorModel."""
        return DombiTSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class ADMTSKClassifier(_BaseClassifierEstimator):
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
        from highfis import ADMTSKClassifier

        clf = ADMTSKClassifier()
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
        epochs: int = 50,
        learning_rate: float = 1e-2,
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
        adaptive: bool = True,
        lambda_: float = 1.0,
        lower_bound: float = 1.0 / math.e,
        k: float = 10.0,
        zero_consequent_init: bool = True,
        device: str = "cpu",
    ) -> None:
        """Initialize an ADMTSK classifier estimator.

        Args:
            input_configs: Optional list of per-feature input configurations.
            n_mfs: Number of membership functions per input when using
                ``mf_init="kmeans"``, ``"minibatch_kmeans"``, or ``"grid"``.
            mf_init: Initialisation strategy for MFs: ``"kmeans"`` (default),
                ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
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
            weight_decay: Weight decay applied during training.
            adaptive: If True, use adaptive lambda selection for Dombi T-norm.
            lambda_: Fixed Dombi parameter when adaptive is False.
            lower_bound: Lower bound used by Composite GMF.
            k: Heuristic constant used to compute adaptive lambda.
            zero_consequent_init: If True (default), initialize
                consequent parameters to zero.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).

        Raises:
            ValueError: If estimator hyperparameters are invalid.
        """
        resolved_n_mfs = n_mfs
        resolved_mf_init = mf_init
        resolved_sigma_scale = sigma_scale
        resolved_rule_base = rule_base

        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init=resolved_mf_init,
            sigma_scale=resolved_sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=resolved_rule_base,
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
        self.adaptive = adaptive
        self.lambda_ = lambda_
        self.lower_bound = lower_bound
        self.k = k
        self.zero_consequent_init = zero_consequent_init

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        use_paper_default = (
            self.input_configs is None
            and isinstance(self.mf_init, str)
            and self.mf_init.lower() == "grid"
            and int(self.n_mfs) == 3
        )
        if use_paper_default:
            feature_names = [f"x{i + 1}" for i in range(x_arr.shape[1])]
            input_mfs = _build_admtsk_default_input_mfs(x_arr.shape[1])
            effective_rule_base = self.rule_base if self.rule_base is not None else "coco"
            return (
                _wrap_composite_gaussian_input_mfs(input_mfs),
                feature_names,
                effective_rule_base,
            )

        input_mfs, feature_names, effective_rule_base = super()._build_input_mfs(x_arr)
        return (
            _wrap_composite_gaussian_input_mfs(input_mfs),
            feature_names,
            effective_rule_base,
        )

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        return ADMTSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            adaptive=bool(self.adaptive),
            lambda_=float(self.lambda_),
            lower_bound=float(self.lower_bound),
            k=self.k,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            zero_consequent_init=bool(self.zero_consequent_init),
        )


class ADMTSKRegressor(_BaseRegressorEstimator):
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
        from highfis import ADMTSKRegressor

        reg = ADMTSKRegressor()
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
        epochs: int = 50,
        learning_rate: float = 1e-2,
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
        adaptive: bool = True,
        lambda_: float = 1.0,
        lower_bound: float = 1.0 / math.e,
        k: float = 10.0,
        zero_consequent_init: bool = True,
        device: str = "cpu",
    ) -> None:
        """Initialize an ADMTSK regressor estimator.

        Args:
            input_configs: Optional list of per-feature input configurations.
            n_mfs: Number of membership functions per input when using
                ``mf_init="kmeans"``, ``"minibatch_kmeans"``, or ``"grid"``.
            mf_init: Initialisation strategy for MFs: ``"kmeans"`` (default),
                ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
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
            weight_decay: Weight decay applied during training.
            adaptive: If True, use adaptive lambda selection for Dombi T-norm.
            lambda_: Fixed Dombi parameter when adaptive is False.
            lower_bound: Lower bound used by Composite GMF.
            k: Heuristic constant used to compute adaptive lambda.
            zero_consequent_init: If True (default), initialize
                consequent parameters to zero.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).

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
            weight_decay=weight_decay,
            device=device,
        )
        self.adaptive = adaptive
        self.lambda_ = lambda_
        self.lower_bound = lower_bound
        self.k = k
        self.zero_consequent_init = zero_consequent_init

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        use_paper_default = (
            self.input_configs is None
            and isinstance(self.mf_init, str)
            and self.mf_init.lower() == "grid"
            and int(self.n_mfs) == 3
        )
        if use_paper_default:
            feature_names = [f"x{i + 1}" for i in range(x_arr.shape[1])]
            input_mfs = _build_admtsk_default_input_mfs(x_arr.shape[1])
            effective_rule_base = self.rule_base if self.rule_base is not None else "coco"
            return (
                _wrap_composite_gaussian_input_mfs(input_mfs),
                feature_names,
                effective_rule_base,
            )

        input_mfs, feature_names, effective_rule_base = super()._build_input_mfs(x_arr)
        return (
            _wrap_composite_gaussian_input_mfs(input_mfs),
            feature_names,
            effective_rule_base,
        )

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        return ADMTSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            adaptive=bool(self.adaptive),
            lambda_=float(self.lambda_),
            lower_bound=float(self.lower_bound),
            k=self.k,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            zero_consequent_init=bool(self.zero_consequent_init),
        )
