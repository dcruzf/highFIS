"""Sklearn-compatible estimators for HDFIS models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..base import BaseTSK
from ..memberships import (
    MembershipFunction,
)
from ..models import (
    HDFISMinClassifierModel,
    HDFISMinRegressorModel,
    HDFISProdClassifierModel,
    HDFISProdRegressorModel,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
    _wrap_dimension_dependent_gaussian_input_mfs,
)


def _resolve_hdfis_paper_strict_config(
    *,
    paper_strict: bool,
    n_mfs: int | None,
    mf_init: str | None,
    rule_base: str | None,
    batch_size: int | None,
) -> tuple[int, str, str | None, int | None]:
    if not paper_strict:
        return (
            5 if n_mfs is None else int(n_mfs),
            "kmeans" if mf_init is None else str(mf_init),
            rule_base,
            512 if batch_size is None else batch_size,
        )

    if n_mfs is not None and int(n_mfs) != 3:
        raise ValueError("paper_strict requires n_mfs=3")
    if mf_init is not None and str(mf_init).lower() != "grid":
        raise ValueError("paper_strict requires mf_init='grid'")
    if rule_base is not None and str(rule_base).lower() != "coco":
        raise ValueError("paper_strict requires rule_base='coco'")
    if batch_size is not None and int(batch_size) != 64:
        raise ValueError("paper_strict requires batch_size=64")

    return 3, "grid", "coco", 64


class HDFISProdClassifier(_BaseClassifierEstimator):
    r"""HDFIS-prod classifier estimator with dimension-dependent Gaussian MFs.

    HDFIS-prod combines the standard product T-norm with a dimension-dependent
    Gaussian membership function (DMF) to avoid numeric underflow in very
    high-dimensional feature spaces while preserving first-order TSK
    consequents.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.

    Example:
        ```python
        from highfis import HDFISProdClassifier

        clf = HDFISProdClassifier()
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int | None = None,
        mf_init: str | None = None,
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        xi: float = 745.0,
        rho: float | None = None,
        device: str = "cpu",
        paper_strict: bool = False,
    ) -> None:
        r"""Initialise an HDFIS-prod classifier estimator.

        Args:
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of MFs/rules for grid/k-means initialisation.
                Defaults to ``5`` in regular mode and ``3`` in
                ``paper_strict`` mode.
            mf_init: MF initialisation strategy. Defaults to ``"kmeans"`` in
                regular mode and ``"grid"`` in ``paper_strict`` mode.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to model
                defaults in regular mode and is fixed to ``"coco"`` in
                ``paper_strict`` mode.
            batch_size: Mini-batch size. Defaults to ``512`` in regular mode
                and ``64`` in ``paper_strict`` mode.
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum number of point-based FRB rules when
                ``rule_base='pfrb'``. ``None`` uses all training samples.
            patience: Early-stopping patience (default ``20``).
                Set to ``None`` to disable early stopping.
            restore_best: Restore best validation weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            xi: Precision constant used to compute the DMF scale exponent
                $\rho$ when *rho* is ``None``. Must be greater than 1.
            rho: Scale exponent for the dimension-dependent Gaussian MF.
                When ``None``, computed as ``1 - log(xi) / log(D)``.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            paper_strict: If ``True``, enforce the paper protocol defaults
                used in HDFIS_2023 experiments (grid init, CoCo rule base,
                3 rules, batch size 64), use strict DMF equation mode, and
                zero-initialize consequents.
        """
        resolved_n_mfs, resolved_mf_init, resolved_rule_base, resolved_batch_size = _resolve_hdfis_paper_strict_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            mf_init=mf_init,
            rule_base=rule_base,
            batch_size=batch_size,
        )
        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init=resolved_mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=resolved_rule_base,
            batch_size=resolved_batch_size,
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
        self.xi = float(xi)
        self.rho = rho
        self.paper_strict = bool(paper_strict)

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        input_mfs, feature_names, effective_rule_base = super()._build_input_mfs(x_arr)
        return (
            _wrap_dimension_dependent_gaussian_input_mfs(
                input_mfs,
                dimension=x_arr.shape[1],
                xi=self.xi,
                rho=self.rho,
                paper_strict_equation=bool(self.paper_strict),
            ),
            feature_names,
            effective_rule_base,
        )

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        return HDFISProdClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            zero_consequent_init=bool(self.paper_strict),
        )


class HDFISProdRegressor(_BaseRegressorEstimator):
    r"""HDFIS-prod regressor estimator with dimension-dependent Gaussian MFs.

    HDFIS-prod combines the standard product T-norm with a dimension-dependent
    Gaussian membership function (DMF) to avoid numeric underflow in very
    high-dimensional feature spaces while preserving first-order TSK
    consequents.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.

    Example:
        ```python
        from highfis import HDFISProdRegressor

        reg = HDFISProdRegressor()
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int | None = None,
        mf_init: str | None = None,
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        xi: float = 745.0,
        rho: float | None = None,
        device: str = "cpu",
        paper_strict: bool = False,
    ) -> None:
        r"""Initialise an HDFIS-prod regressor estimator.

        Args:
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of MFs/rules for grid/k-means initialisation.
                Defaults to ``5`` in regular mode and ``3`` in
                ``paper_strict`` mode.
            mf_init: MF initialisation strategy. Defaults to ``"kmeans"`` in
                regular mode and ``"grid"`` in ``paper_strict`` mode.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to model
                defaults in regular mode and is fixed to ``"coco"`` in
                ``paper_strict`` mode.
            batch_size: Mini-batch size. Defaults to ``512`` in regular mode
                and ``64`` in ``paper_strict`` mode.
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
                Set to ``None`` to disable early stopping.
            restore_best: Restore best validation weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            xi: Precision constant used to compute the DMF scale exponent
                $\rho$ when *rho* is ``None``. Must be greater than 1.
            rho: Scale exponent for the dimension-dependent Gaussian MF.
                When ``None``, computed as ``1 - log(xi) / log(D)``.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            paper_strict: If ``True``, enforce the paper protocol defaults
                used in HDFIS_2023 experiments (grid init, CoCo rule base,
                3 rules, batch size 64), use strict DMF equation mode, and
                zero-initialize consequents.
        """
        resolved_n_mfs, resolved_mf_init, resolved_rule_base, resolved_batch_size = _resolve_hdfis_paper_strict_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            mf_init=mf_init,
            rule_base=rule_base,
            batch_size=batch_size,
        )
        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init=resolved_mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=resolved_rule_base,
            batch_size=resolved_batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            weight_decay=weight_decay,
            device=device,
        )
        self.xi = float(xi)
        self.rho = rho
        self.paper_strict = bool(paper_strict)

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:
        input_mfs, feature_names, effective_rule_base = super()._build_input_mfs(x_arr)
        return (
            _wrap_dimension_dependent_gaussian_input_mfs(
                input_mfs,
                dimension=x_arr.shape[1],
                xi=self.xi,
                rho=self.rho,
                paper_strict_equation=bool(self.paper_strict),
            ),
            feature_names,
            effective_rule_base,
        )

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        return HDFISProdRegressorModel(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            zero_consequent_init=bool(self.paper_strict),
        )


class HDFISMinClassifier(_BaseClassifierEstimator):
    r"""HDFIS-min classifier estimator with minimum T-norm antecedents.

    HDFIS-min freezes antecedent membership parameters and uses a minimum
    T-norm aggregation in the antecedent, so that only consequent parameters
    are optimized during training. This matches the paper's observation that
    minimum-based high-dimensional inference is best handled by fixing the
    antecedent structure and training the rule consequents.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.

    Example:
        ```python
        from highfis import HDFISMinClassifier

        clf = HDFISMinClassifier()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int | None = None,
        mf_init: str | None = None,
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        device: str = "cpu",
        paper_strict: bool = False,
    ) -> None:
        """Initialise an HDFIS-min classifier estimator.

        Args:
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of MFs/rules for grid/k-means initialisation.
                Defaults to ``5`` in regular mode and ``3`` in
                ``paper_strict`` mode.
            mf_init: MF initialisation strategy. Defaults to ``"kmeans"`` in
                regular mode and ``"grid"`` in ``paper_strict`` mode.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to model
                defaults in regular mode and is fixed to ``"coco"`` in
                ``paper_strict`` mode.
            batch_size: Mini-batch size. Defaults to ``512`` in regular mode
                and ``64`` in ``paper_strict`` mode.
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum number of point-based FRB rules when
                ``rule_base='pfrb'``. ``None`` uses all training samples.
            patience: Early-stopping patience (default ``20``).
                Set to ``None`` to disable early stopping.
            restore_best: Restore best validation weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            paper_strict: If ``True``, enforce the paper protocol defaults
                used in HDFIS_2023 experiments (grid init, CoCo rule base,
                3 rules, batch size 64) and zero-initialize consequents.
        """
        resolved_n_mfs, resolved_mf_init, resolved_rule_base, resolved_batch_size = _resolve_hdfis_paper_strict_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            mf_init=mf_init,
            rule_base=rule_base,
            batch_size=batch_size,
        )
        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init=resolved_mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=resolved_rule_base,
            batch_size=resolved_batch_size,
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
        self.paper_strict = bool(paper_strict)

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        return HDFISMinClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            zero_consequent_init=bool(self.paper_strict),
        )


class HDFISMinRegressor(_BaseRegressorEstimator):
    r"""HDFIS-min regressor estimator with minimum T-norm antecedents.

    HDFIS-min freezes antecedent membership parameters and uses a minimum
    T-norm aggregation in the antecedent, so that only consequent parameters
    are optimized during training. This design avoids the nondifferentiability
    of the minimum operator while preserving first-order TSK consequents.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.

    Example:
        ```python
        from highfis import HDFISMinRegressor

        reg = HDFISMinRegressor()
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int | None = None,
        mf_init: str | None = None,
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        device: str = "cpu",
        paper_strict: bool = False,
    ) -> None:
        """Initialise an HDFIS-min regressor estimator.

        Args:
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of MFs/rules for grid/k-means initialisation.
                Defaults to ``5`` in regular mode and ``3`` in
                ``paper_strict`` mode.
            mf_init: MF initialisation strategy. Defaults to ``"kmeans"`` in
                regular mode and ``"grid"`` in ``paper_strict`` mode.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to model
                defaults in regular mode and is fixed to ``"coco"`` in
                ``paper_strict`` mode.
            batch_size: Mini-batch size. Defaults to ``512`` in regular mode
                and ``64`` in ``paper_strict`` mode.
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``).
                Set to ``None`` to disable early stopping.
            restore_best: Restore best validation weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            paper_strict: If ``True``, enforce the paper protocol defaults
                used in HDFIS_2023 experiments (grid init, CoCo rule base,
                3 rules, batch size 64) and zero-initialize consequents.
        """
        resolved_n_mfs, resolved_mf_init, resolved_rule_base, resolved_batch_size = _resolve_hdfis_paper_strict_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            mf_init=mf_init,
            rule_base=rule_base,
            batch_size=batch_size,
        )
        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init=resolved_mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=resolved_rule_base,
            batch_size=resolved_batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            patience=patience,
            restore_best=restore_best,
            weight_decay=weight_decay,
            device=device,
        )
        self.paper_strict = bool(paper_strict)

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        return HDFISMinRegressorModel(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            zero_consequent_init=bool(self.paper_strict),
        )
