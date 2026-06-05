"""Sklearn-compatible estimators for HTSK and vanilla TSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..base import BaseTSK
from ..memberships import (
    MembershipFunction,
)
from ..models import (
    HTSKClassifierModel,
    HTSKRegressorModel,
    TSKClassifierModel,
    TSKRegressorModel,
)
from ..optim._base import BaseTrainer
from ..optim._gradient import GradientTrainer
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)


def _resolve_htsk_paper_strict_config(
    *,
    paper_strict: bool,
    n_mfs: int | None,
    mf_init: str | None,
    sigma_scale: float | str | None,
    rule_base: str | None,
    epochs: int | None,
    learning_rate: float | None,
    batch_size: int | None,
) -> tuple[int, str, float | str, str | None, int, float, int | None]:
    if not paper_strict:
        return (
            3 if n_mfs is None else int(n_mfs),
            "kmeans" if mf_init is None else str(mf_init),
            1.0 if sigma_scale is None else sigma_scale,
            rule_base,
            10 if epochs is None else int(epochs),
            1e-2 if learning_rate is None else float(learning_rate),
            512 if batch_size is None else batch_size,
        )

    if n_mfs is not None and int(n_mfs) != 30:
        raise ValueError("paper_strict requires n_mfs=30")
    if mf_init is not None and str(mf_init).lower() != "kmeans":
        raise ValueError("paper_strict requires mf_init='kmeans'")
    if sigma_scale is not None and float(sigma_scale) != 1.0:
        raise ValueError("paper_strict requires sigma_scale=1.0")
    if rule_base is not None and str(rule_base).lower() != "coco":
        raise ValueError("paper_strict requires rule_base='coco'")
    if epochs is not None and int(epochs) != 200:
        raise ValueError("paper_strict requires epochs=200")
    if learning_rate is not None and float(learning_rate) != 1e-2:
        raise ValueError("paper_strict requires learning_rate=1e-2")
    if batch_size is not None and int(batch_size) != 512:
        raise ValueError("paper_strict requires batch_size=512")

    return 30, "kmeans", 1.0, "coco", 200, 1e-2, 512


class _HTSKPaperStrictTrainer(GradientTrainer):
    """Trainer variant that follows HTSK_2021 protocol details."""

    def fit(
        self,
        model: BaseTSK,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        effective_batch_size = self.batch_size
        if effective_batch_size is not None and int(effective_batch_size) > int(x.shape[0]):
            effective_batch_size = min(int(x.shape[0]), 60)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.learning_rate))
        return model.fit(
            x,
            y,
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            criterion=self.loss,
            optimizer=optimizer,
            batch_size=effective_batch_size,
            shuffle=bool(self.shuffle),
            ur_weight=float(self.ur_weight),
            ur_target=self.ur_target,
            verbose=self.verbose,
            x_val=x_val,
            y_val=y_val,
            patience=self.patience,
            restore_best=bool(self.restore_best),
            weight_decay=float(self.weight_decay),
        )


class HTSKClassifier(_BaseClassifierEstimator):
    r"""HTSK classifier for high-dimensional TSK inference.

    HTSK replaces the standard product t-norm with a geometric mean over
    membership values and performs rule normalization in log-space.

    References:
        Y. Cui, D. Wu and Y. Xu, "Curse of Dimensionality for TSK Fuzzy
        Neural Networks: Explanation and Solutions," 2021 International
        Joint Conference on Neural Networks (IJCNN), Shenzhen, China,
        2021, pp. 1-8, doi: 10.1109/IJCNN52387.2021.9534265.

    Example:
        ```python
        from highfis import HTSKClassifier

        clf = HTSKClassifier()
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
        epochs: int | None = None,
        learning_rate: float | None = None,
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
        """Initialise an HTSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs.
                Defaults to ``3`` in regular mode and ``30`` in
                ``paper_strict`` mode.
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
                Fixed to ``"kmeans"`` in ``paper_strict`` mode.
            sigma_scale: Sigma scale factor. ``1.0`` is recommended for HTSK
                and required in ``paper_strict`` mode.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs.
                Defaults to ``10`` in regular mode and ``200`` in
                ``paper_strict`` mode.
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
                Fixed to ``"coco"`` in ``paper_strict`` mode.
            batch_size: Mini-batch size.
                Defaults to ``512`` in regular mode and is fixed to
                ``512`` in ``paper_strict`` mode.
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by HTSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            paper_strict: If ``True``, enforce HTSK_2021 protocol defaults
                (``n_mfs=30``, ``mf_init='kmeans'``, ``rule_base='coco'``,
                ``epochs=200``, ``learning_rate=0.01``, ``batch_size=512``),
                and use Adam with the paper batch fallback
                ``min(N_t, 60)`` when ``batch_size > N_t``.
        """
        (
            resolved_n_mfs,
            resolved_mf_init,
            resolved_sigma_scale,
            resolved_rule_base,
            resolved_epochs,
            resolved_learning_rate,
            resolved_batch_size,
        ) = _resolve_htsk_paper_strict_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            rule_base=rule_base,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init=resolved_mf_init,
            sigma_scale=resolved_sigma_scale,
            random_state=random_state,
            epochs=resolved_epochs,
            learning_rate=resolved_learning_rate,
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

    def _get_trainer(self) -> BaseTrainer:
        if not self.paper_strict:
            return super()._get_trainer()
        return _HTSKPaperStrictTrainer(
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
        )

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create HTSKClassifierModel."""
        return HTSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class HTSKRegressor(_BaseRegressorEstimator):
    r"""HTSK regressor for high-dimensional TSK inference.

    HTSK replaces the standard product t-norm with a geometric mean over
    membership values and performs rule normalization in log-space.

    References:
        Y. Cui, D. Wu and Y. Xu, "Curse of Dimensionality for TSK Fuzzy
        Neural Networks: Explanation and Solutions," 2021 International
        Joint Conference on Neural Networks (IJCNN), Shenzhen, China,
        2021, pp. 1-8, doi: 10.1109/IJCNN52387.2021.9534265.

    Example:
        ```python
        from highfis import HTSKRegressor

        reg = HTSKRegressor()
        reg.fit(X_train, y_train)
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
        epochs: int | None = None,
        learning_rate: float | None = None,
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
        """Initialise an HTSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs.
                Defaults to ``3`` in regular mode and ``30`` in
                ``paper_strict`` mode.
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
                Fixed to ``"kmeans"`` in ``paper_strict`` mode.
            sigma_scale: Scale factor for sigma initialisation when
                ``mf_init="kmeans"``. ``1.0`` is recommended for HTSK and
                required in ``paper_strict`` mode.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs.
                Defaults to ``10`` in regular mode and ``200`` in
                ``paper_strict`` mode.
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
                Fixed to ``"coco"`` in ``paper_strict`` mode.
            batch_size: Mini-batch size.
                Defaults to ``512`` in regular mode and is fixed to
                ``512`` in ``paper_strict`` mode.
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by HTSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            paper_strict: If ``True``, enforce HTSK_2021 protocol defaults
                (``n_mfs=30``, ``mf_init='kmeans'``, ``rule_base='coco'``,
                ``epochs=200``, ``learning_rate=0.01``, ``batch_size=512``),
                and use Adam with the paper batch fallback
                ``min(N_t, 60)`` when ``batch_size > N_t``.
        """
        (
            resolved_n_mfs,
            resolved_mf_init,
            resolved_sigma_scale,
            resolved_rule_base,
            resolved_epochs,
            resolved_learning_rate,
            resolved_batch_size,
        ) = _resolve_htsk_paper_strict_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            rule_base=rule_base,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init=resolved_mf_init,
            sigma_scale=resolved_sigma_scale,
            random_state=random_state,
            epochs=resolved_epochs,
            learning_rate=resolved_learning_rate,
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

    def _get_trainer(self) -> BaseTrainer:
        if not self.paper_strict:
            return super()._get_trainer()
        return _HTSKPaperStrictTrainer(
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
        )

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create HTSKRegressorModel."""
        return HTSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


# =====================================================================
# Vanilla TSK Estimators  (Takagi & Sugeno, 1985)
# =====================================================================


class TSKClassifier(_BaseClassifierEstimator):
    r"""Vanilla TSK classifier with sum-based rule normalization.

    The vanilla Takagi-Sugeno-Kang inference computes rule firing strengths
    with the product t-norm and normalizes them by their total sum.

    References:
        T. Takagi and M. Sugeno, "Fuzzy identification of systems and
        its applications to modeling and control," in IEEE
        Transactions on Systems, Man, and Cybernetics, vol. SMC-15,
        no. 1, pp. 116-132, Jan.-Feb. 1985,
        doi: 10.1109/TSMC.1985.6313399.

    Example:
        ```python
        from highfis import TSKClassifier

        clf = TSKClassifier(n_mfs=5, random_state=0)
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
        epochs: int | None = None,
        learning_rate: float | None = None,
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
        """Initialise a vanilla TSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
            sigma_scale: Sigma scale factor. Use ``"auto"`` (= ``sqrt(D)``)
                for high-dimensional data to mitigate softmax saturation
                (Cui et al., IJCNN 2021). ``1.0`` is appropriate for low-
                to medium-dimensional problems.
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
            pfrb_max_rules: Maximum point-based FRB rules (unused by TSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            paper_strict: Enforce HTSK_2021 protocol defaults.
        """
        (
            resolved_n_mfs,
            resolved_mf_init,
            resolved_sigma_scale,
            resolved_rule_base,
            resolved_epochs,
            resolved_learning_rate,
            resolved_batch_size,
        ) = _resolve_htsk_paper_strict_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            rule_base=rule_base,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init=resolved_mf_init,
            sigma_scale=resolved_sigma_scale,
            random_state=random_state,
            epochs=resolved_epochs,
            learning_rate=resolved_learning_rate,
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

    def _get_trainer(self) -> BaseTrainer:
        if not self.paper_strict:
            return super()._get_trainer()
        return _HTSKPaperStrictTrainer(
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
        )

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create TSKClassifierModel."""
        return TSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class TSKRegressor(_BaseRegressorEstimator):
    r"""Vanilla TSK regressor with sum-based rule normalization.

    The vanilla Takagi-Sugeno-Kang inference computes rule firing strengths
    with the product t-norm and normalizes them by their total sum.

    References:
        T. Takagi and M. Sugeno, "Fuzzy identification of systems and
        its applications to modeling and control," in IEEE
        Transactions on Systems, Man, and Cybernetics, vol. SMC-15,
        no. 1, pp. 116-132, Jan.-Feb. 1985,
        doi: 10.1109/TSMC.1985.6313399.

    Example:
        ```python
        from highfis import TSKRegressor

        reg = TSKRegressor(n_mfs=30, random_state=0)
        reg.fit(X_train, y_train)
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
        epochs: int | None = None,
        learning_rate: float | None = None,
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
        """Initialise a vanilla TSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
            sigma_scale: Sigma scale factor. Use ``"auto"`` (= ``sqrt(D)``)
                to mitigate softmax saturation on high-dimensional data.
                ``1.0`` is appropriate for low-to-medium-dimensional problems.
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
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            paper_strict: Enforce HTSK_2021 protocol defaults.
        """
        (
            resolved_n_mfs,
            resolved_mf_init,
            resolved_sigma_scale,
            resolved_rule_base,
            resolved_epochs,
            resolved_learning_rate,
            resolved_batch_size,
        ) = _resolve_htsk_paper_strict_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            rule_base=rule_base,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init=resolved_mf_init,
            sigma_scale=resolved_sigma_scale,
            random_state=random_state,
            epochs=resolved_epochs,
            learning_rate=resolved_learning_rate,
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

    def _get_trainer(self) -> BaseTrainer:
        if not self.paper_strict:
            return super()._get_trainer()
        return _HTSKPaperStrictTrainer(
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
        )

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        return TSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )
