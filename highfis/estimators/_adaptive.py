"""Sklearn-compatible estimators for ADATSK and ADPTSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from ..base import BaseTSK
from ..memberships import (
    ADATSKGaussianMF,
    GaussianMF,
    MembershipFunction,
)
from ..models import (
    ADATSKClassifierModel,
    ADATSKRegressorModel,
    ADPTSKClassifierModel,
    ADPTSKRegressorModel,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
    _wrap_gaussian_pimf_input_mfs,
)


def _set_sigma_to_one_and_freeze(mf: MembershipFunction) -> None:
    """Set Gaussian sigma to 1 and freeze it to match ADATSK paper defaults."""
    if not isinstance(mf, GaussianMF):
        return

    target = max(1.0 - float(mf.eps), float(mf.eps))
    target_t = torch.tensor(target, dtype=mf.raw_sigma.dtype, device=mf.raw_sigma.device)
    mf.raw_sigma.data.copy_(torch.log(torch.expm1(target_t)))
    mf.raw_sigma.requires_grad_(False)


def _apply_adatsk_paper_defaults(
    model: BaseTSK,
    x_t: Tensor,
    *,
    freeze_antecedent_in_high_dim: bool,
    high_dim_threshold: int,
) -> None:
    """Apply ADATSK paper-style MF constraints before training starts."""
    for mf_list in model.membership_layer.input_mfs.values():
        for module in cast(nn.ModuleList, mf_list):
            mf = cast(MembershipFunction, module)
            _set_sigma_to_one_and_freeze(mf)

    if freeze_antecedent_in_high_dim and int(x_t.shape[1]) >= int(high_dim_threshold):
        for param in model.membership_layer.parameters():
            param.requires_grad_(False)


def _wrap_adatsk_gaussian_input_mfs(
    input_mfs: Mapping[str, Sequence[MembershipFunction]],
) -> dict[str, list[MembershipFunction]]:
    """Wrap Gaussian antecedents with ADATSK paper-style Gaussian MFs."""
    wrapped: dict[str, list[MembershipFunction]] = {}
    for name, mfs in input_mfs.items():
        wrapped_mfs: list[MembershipFunction] = []
        for mf in mfs:
            if isinstance(mf, GaussianMF):
                mean = float(mf.mean.detach().cpu().item())
                sigma = float(mf.sigma.detach().cpu().item())
                wrapped_mfs.append(ADATSKGaussianMF(mean=mean, sigma=sigma, eps=mf.eps))
            else:
                wrapped_mfs.append(mf)
        wrapped[name] = wrapped_mfs
    return wrapped


class ADPTSKClassifier(_BaseClassifierEstimator):
    r"""TSK classifier with ADP-softmin antecedent and Gaussian PIMF.

    The firing strengths of each rule are computed with the ADP-softmin
    operator, and membership functions are wrapped as Gaussian PIMFs to
    preserve a positive infimum during high-dimensional training.

    Reference:
        Ma, M., Qian, L., Zhang, Y., Fang, Q., & Xue, G. (2025). An
        adaptive double-parameter softmin based Takagi-Sugeno-Kang
        fuzzy system for high-dimensional data. Fuzzy Sets and
        Systems, 521, 109582.
        https://doi.org/10.1016/j.fss.2025.109582

    Example:
        ```python
        from highfis import ADPTSKClassifier

        clf = ADPTSKClassifier()
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
        weight_decay: float = 1e-8,
        kappa: float = 690.0,
        xi: float = 730.0,
        k: float = 1.0,
        eps: float | None = None,
    ) -> None:
        """Initialise an ADPTSK classifier estimator.

        Args:
            input_configs: Optional list of :class:`InputConfig` instances,
                one per feature. Only ``name`` is used when
                ``mf_init="kmeans"``.
            n_mfs: Number of membership functions per feature or k-means
                clusters.
            mf_init: Membership-function initialization strategy:
                ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``,
                or ``"grid"``.
            sigma_scale: Scale factor for Gaussian MF sigma initialization.
            random_state: Seed for k-means and PyTorch weight initialization.
            epochs: Maximum number of training epochs.
            learning_rate: Initial learning rate for the Adam optimizer.
            verbose: Verbosity level for training output.
            rule_base: Rule-base strategy, e.g. ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size. ``None`` uses the full dataset.
            shuffle: Whether to shuffle training samples each epoch.
            ur_weight: Uniform-rule regularization weight.
            ur_target: Target average rule activation for UR.
            consequent_batch_norm: Apply batch normalization to consequent
                linear layers.
            pfrb_max_rules: Maximum rules for point-based FRB when
                ``rule_base="pfrb"``.
            patience: Early-stopping patience. ``None`` disables early stopping.
            restore_best: Restore the best validation model weights after
                training.
                stopping.
            weight_decay: L2 weight decay coefficient for consequent parameters.
            kappa: ADPTSK ``κ`` parameter controlling the double-softmin
                geometry.
            xi: ADPTSK ``ξ`` parameter controlling adaptive softmin sharpness.
            k: Gaussian PIMF scaling constant used when wrapping the input MFs.
            eps: Optional lower bound for Gaussian PIMF values.
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
        self.kappa = float(kappa)
        self.xi = float(xi)
        self.k = float(k)
        self.eps = eps

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        input_mfs = _wrap_gaussian_pimf_input_mfs(input_mfs, k=self.k, eps=self.eps)
        return ADPTSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            kappa=self.kappa,
            xi=self.xi,
            eps=self.eps,
        )


class ADPTSKRegressor(_BaseRegressorEstimator):
    r"""TSK regressor with ADP-softmin antecedent and Gaussian PIMF.

    The firing strengths of each rule are computed with the ADP-softmin
    operator, and membership functions are wrapped as Gaussian PIMFs to
    preserve a positive infimum during high-dimensional training.

    Reference:
        Ma, M., Qian, L., Zhang, Y., Fang, Q., & Xue, G. (2025). An
        adaptive double-parameter softmin based Takagi-Sugeno-Kang
        fuzzy system for high-dimensional data. Fuzzy Sets and
        Systems, 521, 109582.
        https://doi.org/10.1016/j.fss.2025.109582

    Example:
        ```python
        from highfis import ADPTSKRegressor

        reg = ADPTSKRegressor()
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
        weight_decay: float = 1e-8,
        kappa: float = 690.0,
        xi: float = 730.0,
        k: float = 1.0,
        eps: float | None = None,
    ) -> None:
        """Initialise an ADPTSK regressor estimator.

        Args:
            input_configs: Optional list of :class:`InputConfig` instances,
                one per feature. Only ``name`` is used when
                ``mf_init="kmeans"``.
            n_mfs: Number of membership functions per feature or k-means
                clusters.
            mf_init: Membership-function initialization strategy:
                ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``,
                or ``"grid"``.
            sigma_scale: Scale factor for Gaussian MF sigma initialization.
            random_state: Seed for k-means and PyTorch weight initialization.
            epochs: Maximum number of training epochs.
            learning_rate: Initial learning rate for the Adam optimizer.
            verbose: Verbosity level for training output.
            rule_base: Rule-base strategy, e.g. ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size. ``None`` uses the full dataset.
            shuffle: Whether to shuffle training samples each epoch.
            ur_weight: Uniform-rule regularization weight.
            ur_target: Target average rule activation for UR.
            consequent_batch_norm: Apply batch normalization to consequent
                linear layers.
            pfrb_max_rules: Maximum rules for point-based FRB when
                ``rule_base="pfrb"``.
            patience: Early-stopping patience. ``None`` disables early stopping.
            restore_best: Restore the best validation model weights after
                training.
                stopping.
            weight_decay: L2 weight decay coefficient for consequent parameters.
            kappa: ADPTSK ``κ`` parameter controlling the double-softmin
                geometry.
            xi: ADPTSK ``ξ`` parameter controlling adaptive softmin sharpness.
            k: Gaussian PIMF scaling constant used when wrapping the input MFs.
            eps: Optional lower bound for Gaussian PIMF values.
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
        self.kappa = float(kappa)
        self.xi = float(xi)
        self.k = float(k)
        self.eps = eps

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        input_mfs = _wrap_gaussian_pimf_input_mfs(input_mfs, k=self.k, eps=self.eps)
        return ADPTSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            kappa=self.kappa,
            xi=self.xi,
            eps=self.eps,
        )


class ADATSKClassifier(_BaseClassifierEstimator):
    r"""TSK classifier with adaptive softmin antecedent (ADATSK).

    The firing strength of each rule is computed with the Ada-softmin operator.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import ADATSKClassifier

        clf = ADATSKClassifier(n_mfs=30, random_state=0)
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
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = "coco",
        batch_size: int | None = None,
        shuffle: bool = False,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = None,
        restore_best: bool = False,
        weight_decay: float = 0.0,
        freeze_antecedent_in_high_dim: bool = True,
        high_dim_threshold: int = 1000,
    ) -> None:
        """Initialise an ADATSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of MFs per feature (default ``3``).
            mf_init: MF initialization strategy (default ``"grid"`` for paper-style
                evenly spaced centers). Other options: ``"kmeans"``,
                ``"minibatch_kmeans"``, ``"fcm"``.
            sigma_scale: Sigma scale factor used by clustering initializers.
                In paper-strict mode, Gaussian sigmas are reset to ``1`` and frozen.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: Rule-base strategy. Default ``"coco"`` to match the paper.
            batch_size: Mini-batch size. Default ``None`` (full-batch GD).
            shuffle: Whether to reshuffle each epoch. Default ``False``.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience. Default ``None`` (disabled).
            restore_best: Restore best validation weights. Default ``False``.
            weight_decay: L2 weight decay for consequent parameters. Default ``0.0``.
            freeze_antecedent_in_high_dim: If ``True`` (default), freeze
                antecedent parameters when ``n_features >= high_dim_threshold``.
            high_dim_threshold: Feature-count threshold used to trigger
                antecedent freezing (default ``1000``).
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
        self.freeze_antecedent_in_high_dim = bool(freeze_antecedent_in_high_dim)
        self.high_dim_threshold = int(high_dim_threshold)

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create ADATSKClassifierModel."""
        input_mfs = _wrap_adatsk_gaussian_input_mfs(input_mfs)
        return ADATSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            paper_zero_consequent_init=True,
        )

    def _resolve_input_configs(self, x: np.ndarray) -> list[InputConfig]:
        """Use paper-style grid defaults for ADATSKClassifier.

        The paper places centers on ``[Vmin, Vmax]`` with no range padding.
        This is applied only when configs are auto-generated by defaults.
        """
        configs = super()._resolve_input_configs(x)
        if self.input_configs is not None:
            return configs
        return [InputConfig(name=cfg.name, n_mfs=cfg.n_mfs, overlap=cfg.overlap, margin=0.0) for cfg in configs]

    def _pre_train_hook(self, model: BaseTSK, x_t: Tensor, y_t: Tensor) -> None:
        _apply_adatsk_paper_defaults(
            model,
            x_t,
            freeze_antecedent_in_high_dim=self.freeze_antecedent_in_high_dim,
            high_dim_threshold=self.high_dim_threshold,
        )


class ADATSKRegressor(_BaseRegressorEstimator):
    r"""TSK regressor with adaptive softmin antecedent (ADATSK).

    The firing strength of each rule is computed with the Ada-softmin operator.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import ADATSKRegressor

        reg = ADATSKRegressor(n_mfs=30, random_state=0)
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
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = "coco",
        batch_size: int | None = None,
        shuffle: bool = False,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = None,
        restore_best: bool = False,
        weight_decay: float = 0.0,
        freeze_antecedent_in_high_dim: bool = True,
        high_dim_threshold: int = 1000,
    ) -> None:
        """Initialise an ADATSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of MFs per feature (default ``3``).
            mf_init: MF initialization strategy (default ``"grid"``).
            sigma_scale: Sigma scale factor used by clustering initializers.
                In paper-style mode, Gaussian sigmas are reset to ``1`` and frozen.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: Rule-base strategy. Default ``"coco"``.
            batch_size: Mini-batch size. Default ``None`` (full-batch GD).
            shuffle: Whether to reshuffle each epoch. Default ``False``.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience. Default ``None`` (disabled).
            restore_best: Restore best validation weights. Default ``False``.
            weight_decay: L2 weight decay for consequent parameters. Default ``0.0``.
            freeze_antecedent_in_high_dim: If ``True`` (default), freeze
                antecedent parameters when ``n_features >= high_dim_threshold``.
            high_dim_threshold: Feature-count threshold used to trigger
                antecedent freezing (default ``1000``).
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
        self.freeze_antecedent_in_high_dim = bool(freeze_antecedent_in_high_dim)
        self.high_dim_threshold = int(high_dim_threshold)

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create ADATSKRegressorModel."""
        return ADATSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )

    def _pre_train_hook(self, model: BaseTSK, x_t: Tensor, y_t: Tensor) -> None:
        _apply_adatsk_paper_defaults(
            model,
            x_t,
            freeze_antecedent_in_high_dim=self.freeze_antecedent_in_high_dim,
            high_dim_threshold=self.high_dim_threshold,
        )
