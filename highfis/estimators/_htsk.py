"""Sklearn-compatible estimators for HTSK and vanilla TSK models."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..defuzzifiers import resolve_defuzzifier
from ..memberships import (
    BellMF,
    GaussianMF,
    GaussianPiMF,
    MembershipFunction,
    TrapezoidalMF,
    TriangularMF,
)
from ..models import (
    BaseTSK,
    HTSKClassifierModel,
    HTSKRegressorModel,
    TSKClassifierModel,
    TSKRegressorModel,
)
from ..t_norms import resolve_t_norm
from ._base import (
    BatchSizeSpec,
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)

# Half width at half maximum of a Gaussian, in units of its sigma.
_GAUSSIAN_HWHM = math.sqrt(2.0 * math.log(2.0))

# Membership functions the generic TSK estimators can build. Initialization always yields a
# centre and a Gaussian width. Bell, triangular and trapezoidal sets are placed on the same
# centre with the same width at half maximum, so changing ``mf`` changes the shape and nothing
# else; ``gaussian_pi`` reuses the mean and sigma, since it is the Gaussian with a lower bound.
_MF_BUILDERS: dict[str, Callable[[float, float], MembershipFunction]] = {
    "gaussian": lambda c, s: GaussianMF(mean=c, sigma=s),
    "gaussian_pi": lambda c, s: GaussianPiMF(mean=c, sigma=s),
    "bell": lambda c, s: BellMF(a=_GAUSSIAN_HWHM * s, center=c),
    "triangular": lambda c, s: TriangularMF(
        left=c - 2.0 * _GAUSSIAN_HWHM * s, center=c, right=c + 2.0 * _GAUSSIAN_HWHM * s
    ),
    "trapezoidal": lambda c, s: TrapezoidalMF(
        a=c - 1.5 * _GAUSSIAN_HWHM * s,
        b=c - 0.5 * _GAUSSIAN_HWHM * s,
        c=c + 0.5 * _GAUSSIAN_HWHM * s,
        d=c + 1.5 * _GAUSSIAN_HWHM * s,
    ),
}


def _convert_input_mfs(
    input_mfs: Mapping[str, Sequence[MembershipFunction]], mf: str
) -> Mapping[str, Sequence[MembershipFunction]]:
    """Replace the Gaussian sets produced by initialization with sets of type *mf*.

    Sets that are not plain Gaussians are kept as they are: that is the case when a saved
    estimator is reloaded, whose membership functions already have the chosen type.
    """
    if mf not in _MF_BUILDERS:
        raise ValueError(f"mf must be one of {sorted(_MF_BUILDERS)}; got {mf!r}")
    if mf == "gaussian":
        return input_mfs
    build = _MF_BUILDERS[mf]
    return {
        name: [
            build(float(m.mean.detach().cpu().item()), float(m.sigma.detach().cpu().item()))
            if type(m) is GaussianMF
            else m
            for m in mfs
        ]
        for name, mfs in input_mfs.items()
    }


def _htsk_paper_batch_size(n_samples: int) -> int | None:
    """HTSK_2021 batching: 512, clamped to ``min(N, 60)`` when it exceeds the training set.

    The paper pairs the 512 with a mandatory clamp; without it the batch silently becomes
    full-batch on the small datasets these models target, giving one update per epoch.
    """
    if n_samples < 512:
        return min(n_samples, 60)
    return 512


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
        n_mfs: int = 3,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: BatchSizeSpec = "auto",
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
    ) -> None:
        """Initialise an HTSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs. (default ``3``)
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` is recommended for HTSK.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs. (default ``100``).
            learning_rate: AdamW learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size. ``"auto"`` (default) follows the source article:
                512, or ``min(N, 60)`` when the training set is smaller than that. An integer sets the size and
                ``None`` trains on the full batch.
            shuffle: Reshuffle each epoch.
            ur_weight: Weight of the uniform regularization (UR) term, a penalty on the deviation of
                each rule's average normalized firing strength from ``ur_target`` (Cui, Wu and Huang,
                2020). ``0.0`` disables the penalty.
            ur_target: Target average normalized firing strength of each rule. ``None`` uses ``1/R``,
                where ``R`` is the number of rules.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by HTSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
                Early stopping needs a validation set passed to ``fit``; without one this has
                no effect.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
                Has no effect unless a validation set is passed to ``fit``.
            weight_decay: L2 weight decay for consequent parameters.
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
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
        )

    def _paper_batch_size(self, n_samples: int) -> int | None:
        """HTSK_2021: 512, clamped to ``min(N, 60)`` on smaller training sets."""
        return _htsk_paper_batch_size(n_samples)

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create HTSKClassifierModel."""
        return HTSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            rules=rules,
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
        n_mfs: int = 3,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: BatchSizeSpec = "auto",
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
    ) -> None:
        """Initialise an HTSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs. (default ``3``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
            sigma_scale: Scale factor for sigma initialisation when
                ``mf_init="kmeans"``. ``1.0`` is recommended for HTSK.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs. (default ``100``).
            learning_rate: AdamW learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size. ``"auto"`` (default) follows the source article:
                512, or ``min(N, 60)`` when the training set is smaller than that. An integer sets the size and
                ``None`` trains on the full batch.
            shuffle: Reshuffle each epoch.
            ur_weight: Weight of the uniform regularization (UR) term, a penalty on the deviation of
                each rule's average normalized firing strength from ``ur_target`` (Cui, Wu and Huang,
                2020). ``0.0`` disables the penalty.
            ur_target: Target average normalized firing strength of each rule. ``None`` uses ``1/R``,
                where ``R`` is the number of rules.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by HTSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
                Early stopping needs a validation set passed to ``fit``; without one this has
                no effect.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
                Has no effect unless a validation set is passed to ``fit``.
            weight_decay: L2 weight decay for consequent parameters.
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
        resolved_n_mfs = 3 if n_mfs is None else n_mfs
        resolved_mf_init = "kmeans" if mf_init is None else mf_init
        resolved_sigma_scale = 1.0 if sigma_scale is None else sigma_scale
        resolved_rule_base = rule_base
        resolved_epochs = 100 if epochs is None else epochs
        resolved_learning_rate = 1e-2 if learning_rate is None else learning_rate
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
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
        )

    def _paper_batch_size(self, n_samples: int) -> int | None:
        """HTSK_2021: 512, clamped to ``min(N, 60)`` on smaller training sets."""
        return _htsk_paper_batch_size(n_samples)

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create HTSKRegressorModel."""
        return HTSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            rules=rules,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


# =====================================================================
# Vanilla TSK Estimators  (Takagi & Sugeno, 1985)
# =====================================================================


class TSKClassifier(_BaseClassifierEstimator):
    r"""Generic TSK classifier; by default the classical (vanilla) system.

    The classical Takagi-Sugeno-Kang inference uses Gaussian membership functions,
    computes rule firing strengths with the product t-norm and normalizes them by
    their total sum. That is the default. The three building blocks can be changed
    one at a time through ``mf``, ``t_norm`` and ``defuzzifier``; for example the
    geometric mean with the softmax-in-log defuzzifier gives HTSK.

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
        n_mfs: int = 3,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        mf: str = "gaussian",
        t_norm: str = "prod",
        defuzzifier: str = "sum",
        random_state: int | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: BatchSizeSpec = "auto",
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
    ) -> None:
        """Initialise a vanilla TSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``3``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
            sigma_scale: Sigma scale factor. Use ``"auto"`` (= ``sqrt(D)``)
                for high-dimensional data to mitigate softmax saturation
                (Cui et al., IJCNN 2021). ``1.0`` is appropriate for low-
                to medium-dimensional problems.
            mf: Shape of the membership functions: ``"gaussian"`` (default),
                ``"gaussian_pi"``, ``"bell"``, ``"triangular"`` or ``"trapezoidal"``. Every
                shape is placed on the centre found by ``mf_init``. Bell, triangular and
                trapezoidal sets get the same width at half maximum as the Gaussian;
                ``"gaussian_pi"`` keeps the Gaussian's mean and sigma and adds a positive
                lower bound.
            t_norm: T-norm that aggregates the membership degrees of a rule: ``"prod"``
                (default), ``"min"``, ``"gmean"``, ``"dombi"``, ``"yager"``,
                ``"yager_simple"`` or ``"ale_softmin_yager"``.
            defuzzifier: Normalization of the rule firing strengths: ``"sum"`` (default),
                ``"softmax_log"``, ``"log_sum"`` or ``"inv_log"``.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``100``).
            learning_rate: AdamW learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size. ``"auto"`` (default) follows the source article:
                512, or ``min(N, 60)`` when the training set is smaller than that. An integer sets the size and
                ``None`` trains on the full batch.
            shuffle: Reshuffle each epoch.
            ur_weight: Weight of the uniform regularization (UR) term, a penalty on the deviation of
                each rule's average normalized firing strength from ``ur_target`` (Cui, Wu and Huang,
                2020). ``0.0`` disables the penalty.
            ur_target: Target average normalized firing strength of each rule. ``None`` uses ``1/R``,
                where ``R`` is the number of rules.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum point-based FRB rules (unused by TSK).
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
                Early stopping needs a validation set passed to ``fit``; without one this has
                no effect.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
                Has no effect unless a validation set is passed to ``fit``.
            weight_decay: L2 weight decay for consequent parameters.
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
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
        )
        self.mf = mf
        self.t_norm = t_norm
        self.defuzzifier = defuzzifier

    def _paper_batch_size(self, n_samples: int) -> int | None:
        """HTSK_2021: 512, clamped to ``min(N, 60)`` on smaller training sets."""
        return _htsk_paper_batch_size(n_samples)

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create TSKClassifierModel."""
        return TSKClassifierModel(
            _convert_input_mfs(input_mfs, self.mf),
            n_classes=n_classes,
            rule_base=rule_base,
            t_norm=resolve_t_norm(self.t_norm),
            rules=rules,
            defuzzifier=resolve_defuzzifier(self.defuzzifier),
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class TSKRegressor(_BaseRegressorEstimator):
    r"""Generic TSK regressor; by default the classical (vanilla) system.

    The classical Takagi-Sugeno-Kang inference uses Gaussian membership functions,
    computes rule firing strengths with the product t-norm and normalizes them by
    their total sum. That is the default. The three building blocks can be changed
    one at a time through ``mf``, ``t_norm`` and ``defuzzifier``; for example the
    geometric mean with the softmax-in-log defuzzifier gives HTSK.

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
        n_mfs: int = 3,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        mf: str = "gaussian",
        t_norm: str = "prod",
        defuzzifier: str = "sum",
        random_state: int | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
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
        """Initialise a vanilla TSK regressor.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``3``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
            sigma_scale: Sigma scale factor. Use ``"auto"`` (= ``sqrt(D)``)
                to mitigate softmax saturation on high-dimensional data.
                ``1.0`` is appropriate for low-to-medium-dimensional problems.
            mf: Shape of the membership functions: ``"gaussian"`` (default),
                ``"gaussian_pi"``, ``"bell"``, ``"triangular"`` or ``"trapezoidal"``. Every
                shape is placed on the centre found by ``mf_init``. Bell, triangular and
                trapezoidal sets get the same width at half maximum as the Gaussian;
                ``"gaussian_pi"`` keeps the Gaussian's mean and sigma and adds a positive
                lower bound.
            t_norm: T-norm that aggregates the membership degrees of a rule: ``"prod"``
                (default), ``"min"``, ``"gmean"``, ``"dombi"``, ``"yager"``,
                ``"yager_simple"`` or ``"ale_softmin_yager"``.
            defuzzifier: Normalization of the rule firing strengths: ``"sum"`` (default),
                ``"softmax_log"``, ``"log_sum"`` or ``"inv_log"``.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``100``).
            learning_rate: AdamW learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``. Defaults to
                ``"coco"`` for kmeans and ``"cartesian"`` for grid.
            batch_size: Mini-batch size. ``"auto"`` (default) follows the source article:
                512, or ``min(N, 60)`` when the training set is smaller than that. An integer sets the size and
                ``None`` trains on the full batch.
            shuffle: Reshuffle each epoch.
            ur_weight: Weight of the uniform regularization (UR) term, a penalty on the deviation of
                each rule's average normalized firing strength from ``ur_target`` (Cui, Wu and Huang,
                2020). ``0.0`` disables the penalty.
            ur_target: Target average normalized firing strength of each rule. ``None`` uses ``1/R``,
                where ``R`` is the number of rules.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
                Early stopping needs a validation set passed to ``fit``; without one this has
                no effect.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
                Has no effect unless a validation set is passed to ``fit``.
            weight_decay: L2 weight decay for consequent parameters.
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
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
        )
        self.mf = mf
        self.t_norm = t_norm
        self.defuzzifier = defuzzifier

    def _paper_batch_size(self, n_samples: int) -> int | None:
        """HTSK_2021: 512, clamped to ``min(N, 60)`` on smaller training sets."""
        return _htsk_paper_batch_size(n_samples)

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        return TSKRegressorModel(
            _convert_input_mfs(input_mfs, self.mf),
            rule_base=rule_base,
            t_norm=resolve_t_norm(self.t_norm),
            rules=rules,
            defuzzifier=resolve_defuzzifier(self.defuzzifier),
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )
