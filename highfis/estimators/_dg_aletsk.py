"""Sklearn-compatible estimators for DG-ALETSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from torch import Tensor

from ..memberships import MembershipFunction
from ..models import (
    BaseTSK,
    DGALETSKClassifierModel,
    DGALETSKRegressorModel,
)
from ..optim._base import BaseTrainer
from ..optim._dg import DGTrainer
from ..optim._protocols import PFRBModelProtocol
from ._base import InputConfig
from ._fsre import (
    FSREADATSKClassifier,
    FSREADATSKRegressor,
)

_DG_ALETSK_PAPER_ZETA_GRID: list[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


class DGALETSKClassifier(FSREADATSKClassifier):
    """DG-ALETSK classifier with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-ADATSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.  Training follows the three-phase DG protocol:

    1. **DG phase** — Train only the feature/rule gate parameters and the
       zero-order consequents; the antecedent MFs are frozen (paper Section
       III-C: "we only optimize the gate parameters and the consequents in
       the DG phase"). The zero-order consequent bias is initialised from the
       target labels (Eq. 25).
    2. **Threshold search** — Grid-search for the pruning thresholds
       ``(zeta_lambda, zeta_theta)`` that maximise held-out accuracy.
    3. **Fine-tune phase** — Convert to first-order rules (carrying over the
       label-initialised consequent bias) and optimise all parameters
       (centres, spreads, and consequents) by default, matching the paper.

    Reference:
        G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional
        Fuzzy Approach With Simultaneous Feature Selection and Rule
        Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no.
        11, pp. 3866-3880, Nov. 2023, doi: 10.1109/TFUZZ.2023.3270445.

    Example:
        ```python
        from highfis import DGALETSKClassifier

        clf = DGALETSKClassifier(n_mfs=30, random_state=0)
        clf.fit(X_train, y_train, x_val=X_val, y_val=y_val)
        ```
    """

    def __init__(
        self,
        *,
        lambda_init: float = 1.0,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        dg_epochs: int = 10,
        finetune_epochs: int = 50,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = "pfrb",
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        zeta_lambda: list[float] | None = None,
        zeta_theta: list[float] | None = None,
        use_lse: bool = False,
        trainer: BaseTrainer | None = None,
        optimizer_type: str = "adam",
        structural_pruning: bool = True,
        freeze_antecedents_finetune: bool = False,
        device: str = "cpu",
        eval_metrics_every: int = 1,
        scheduler_class: type[Any] | None = None,
        scheduler_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialise a DG-ALETSK classifier.

        Args:
            lambda_init: Accepted for API compatibility (not used by
                DG-ALETSK; see :class:`FSREADATSKClassifier`).
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB).
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``,
                or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            dg_epochs: Maximum epochs for phase 1 (DG training). Default
                ``10`` follows the paper.
            finetune_epochs: Maximum epochs for phase 3 (fine-tune).
                Default ``50`` follows the DG-ALETSK paper setup.
            learning_rate: Adam learning rate for both phases.
            verbose: Print per-epoch progress.
            rule_base: ``"coco"``, ``"cartesian"``, or ``"pfrb"``.
                Default ``"pfrb"`` follows the paper workflow.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum number of point-based FRB rules when
                ``rule_base='pfrb'``. If ``None`` (default), uses
                paper-style automatic cap: ``100`` rules (or ``50`` when
                ``D >= 10000``).
            patience: Early-stopping patience.  ``None`` disables.
            restore_best: Restore best validation weights after fine-tuning.
            weight_decay: L2 weight decay for consequent parameters.
            zeta_lambda: Grid of λ-pruning threshold candidates. Default
                follows the paper: ``[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]``.
            zeta_theta: Grid of θ-pruning threshold candidates. Default
                follows the paper: ``[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]``.
            use_lse: Refit first-order consequents via LSE during threshold
                search. Default ``False`` for the classification path.
            trainer: Optional custom :class:`~highfis.optim.BaseTrainer`.
                When ``None`` (default) a :class:`~highfis.optim.DGTrainer`
                is built from this estimator's hyperparameters.
            optimizer_type: Optimizer type. ``"adam"`` (default) matches the
                DG-ALETSK paper (Section IV). Also accepts ``"sgd"`` and
                ``"adamw"``; any other value raises ``ValueError``.
            structural_pruning: If ``True`` (default), apply hard structural
                pruning after threshold search.
            freeze_antecedents_finetune: If ``False`` (default), optimise the
                antecedent MF parameters (centres and spreads) during
                fine-tuning, matching the paper. Feature gates are always kept
                frozen during fine-tuning. Set ``True`` to freeze the MFs too.
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
            lambda_init=lambda_init,
            use_en_frb=use_en_frb,
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            fs_epochs=dg_epochs,
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
            trainer=trainer,
            device=device,
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
        )
        self.dg_epochs = dg_epochs
        self.use_lse = use_lse
        self.optimizer_type = optimizer_type
        self.freeze_antecedents_finetune = freeze_antecedents_finetune
        self.finetune_epochs = finetune_epochs
        self.structural_pruning = structural_pruning
        self.zeta_lambda: list[float] | None = zeta_lambda
        self.zeta_theta: list[float] | None = zeta_theta
        self.pfrb_max_rules = pfrb_max_rules

    @staticmethod
    def _resolve_default_pfrb_max_rules(n_features: int) -> int:
        return 50 if int(n_features) >= 10_000 else 100

    def _build_input_mfs(self, x_arr: np.ndarray):
        # Delegate to base behavior, but enforce paper-style P-FRB cap when unset.
        if self.rule_base == "pfrb" and self.pfrb_max_rules is None:
            original = self.pfrb_max_rules
            self.pfrb_max_rules = self._resolve_default_pfrb_max_rules(int(x_arr.shape[1]))
            try:
                return super()._build_input_mfs(x_arr)
            finally:
                self.pfrb_max_rules = original
        return super()._build_input_mfs(x_arr)

    def _effective_pfrb_max_rules(self, n_features: int) -> int | None:
        if self.pfrb_max_rules is None:
            return self._resolve_default_pfrb_max_rules(n_features)
        return self.pfrb_max_rules

    def _pre_train_hook(self, model: BaseTSK, x_t: Tensor, y_t: Tensor) -> None:
        if self.rule_base == "pfrb" and hasattr(model, "init_consequents_from_labels"):
            cast(PFRBModelProtocol, model).init_consequents_from_labels(self._pfrb_aligned_labels(x_t, y_t))

    def _select_model_features(self, x_arr: np.ndarray) -> np.ndarray:
        """Slice inputs to the surviving features when structural pruning shrank the model."""
        from ._dg_tsk import _select_dgtsking_surviving_features

        return _select_dgtsking_surviving_features(self, x_arr)

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create DGALETSKClassifierModel."""
        return DGALETSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            rules=rules,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
            optimizer_type=self.optimizer_type,
        )

    def _get_trainer(self) -> DGTrainer:
        """Return a :class:`~highfis.optim.DGTrainer` built from this estimator's params."""
        zl = self.zeta_lambda if self.zeta_lambda is not None else _DG_ALETSK_PAPER_ZETA_GRID
        zt = self.zeta_theta if self.zeta_theta is not None else _DG_ALETSK_PAPER_ZETA_GRID
        resolved_zeta_lambda = [float(v) for v in zl]
        resolved_zeta_theta = [float(v) for v in zt]
        return DGTrainer(
            dg_epochs=int(self.dg_epochs),
            dg_learning_rate=float(self.learning_rate),
            dg_batch_size=self._effective_batch_size,
            dg_shuffle=bool(self.shuffle),
            dg_patience=self.patience,
            dg_weight_decay=float(self.weight_decay),
            dg_ur_weight=float(self.ur_weight),
            dg_ur_target=self.ur_target,
            zeta_lambda=resolved_zeta_lambda,
            zeta_theta=resolved_zeta_theta,
            use_lse=bool(self.use_lse),
            finetune_epochs=int(self.finetune_epochs),
            finetune_learning_rate=float(self.learning_rate),
            finetune_batch_size=self._effective_batch_size,
            finetune_shuffle=bool(self.shuffle),
            finetune_patience=self.patience,
            finetune_restore_best=bool(self.restore_best),
            finetune_weight_decay=float(self.weight_decay),
            finetune_ur_weight=float(self.ur_weight),
            finetune_ur_target=self.ur_target,
            verbose=self.verbose,
            eval_metrics_every=self.eval_metrics_every,
            scheduler_class=self.scheduler_class,
            scheduler_params=self.scheduler_params,
            optimizer_type=self.optimizer_type,
            structural_pruning=bool(self.structural_pruning),
            finetune_freeze_antecedents=bool(self.freeze_antecedents_finetune),
        )

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
        metrics: list[str] | None = None,
    ) -> DGALETSKClassifier:
        """Train the DG-ALETSK classifier.

        Validation data should be supplied using ``x_val`` and ``y_val``
        when available.
        """
        return cast(DGALETSKClassifier, super().fit(x, y, x_val=x_val, y_val=y_val, metrics=metrics))


class DGALETSKRegressor(FSREADATSKRegressor):
    """DG-ALETSK regressor with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-ADATSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.  Training follows the three-phase DG protocol.

    Reference:
        G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional
        Fuzzy Approach With Simultaneous Feature Selection and Rule
        Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no.
        11, pp. 3866-3880, Nov. 2023, doi: 10.1109/TFUZZ.2023.3270445.

    Example:
        ```python
        from highfis import DGALETSKRegressor

        reg = DGALETSKRegressor(n_mfs=30, random_state=0)
        reg.fit(X_train, y_train, x_val=X_val, y_val=y_val)
        ```
    """

    def __init__(
        self,
        *,
        lambda_init: float = 1.0,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        dg_epochs: int = 10,
        finetune_epochs: int = 200,
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
        zeta_lambda: list[float] | None = None,
        zeta_theta: list[float] | None = None,
        use_lse: bool = True,
        trainer: BaseTrainer | None = None,
        optimizer_type: str = "adam",
        structural_pruning: bool = True,
        freeze_antecedents_finetune: bool = False,
        device: str = "cpu",
        eval_metrics_every: int = 1,
        scheduler_class: type[Any] | None = None,
        scheduler_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialise a DG-ALETSK regressor.

        Args:
            lambda_init: Accepted for API compatibility (not used by
                DG-ALETSK; see :class:`FSREADATSKRegressor`).
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB).
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``,
                or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            dg_epochs: Maximum epochs for phase 1 (DG training). Default
                ``10`` follows the paper.
            finetune_epochs: Maximum epochs for phase 3 (fine-tune).
            learning_rate: Adam learning rate for both phases.
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience.  ``None`` disables.
            restore_best: Restore best validation weights after fine-tuning.
            weight_decay: L2 weight decay for consequent parameters.
            zeta_lambda: Grid of λ-pruning threshold candidates.
            zeta_theta: Grid of θ-pruning threshold candidates.
            use_lse: Refit first-order consequents via LSE during threshold
                search (default ``True``).
            trainer: Optional custom :class:`~highfis.optim.BaseTrainer`.
                When ``None`` (default) a :class:`~highfis.optim.DGTrainer`
                is built from this estimator's hyperparameters.
            optimizer_type: Optimizer type. ``"adam"`` (default) matches the
                DG-ALETSK paper (Section IV). Also accepts ``"sgd"`` and
                ``"adamw"``; any other value raises ``ValueError``.
            structural_pruning: If ``True`` (default), apply hard structural
                pruning after threshold search.
            freeze_antecedents_finetune: If ``False`` (default), optimise the
                antecedent MF parameters (centres and spreads) during
                fine-tuning, matching the paper. Feature gates are always kept
                frozen during fine-tuning. Set ``True`` to freeze the MFs too.
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
            lambda_init=lambda_init,
            use_en_frb=use_en_frb,
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            fs_epochs=dg_epochs,
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
            trainer=trainer,
            device=device,
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
        )
        self.dg_epochs = dg_epochs
        self.use_lse = use_lse
        self.optimizer_type = optimizer_type
        self.freeze_antecedents_finetune = freeze_antecedents_finetune
        self.finetune_epochs = finetune_epochs
        self.structural_pruning = structural_pruning
        self.zeta_lambda: list[float] | None = zeta_lambda
        self.zeta_theta: list[float] | None = zeta_theta

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create DGALETSKRegressorModel."""
        return DGALETSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            rules=rules,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
            optimizer_type=self.optimizer_type,
        )

    def _get_trainer(self) -> DGTrainer:
        """Return a :class:`~highfis.optim.DGTrainer` built from this estimator's params."""
        zl = self.zeta_lambda if self.zeta_lambda is not None else _DG_ALETSK_PAPER_ZETA_GRID
        zt = self.zeta_theta if self.zeta_theta is not None else _DG_ALETSK_PAPER_ZETA_GRID
        resolved_zeta_lambda = [float(v) for v in zl]
        resolved_zeta_theta = [float(v) for v in zt]
        return DGTrainer(
            dg_epochs=int(self.dg_epochs),
            dg_learning_rate=float(self.learning_rate),
            dg_batch_size=self._effective_batch_size,
            dg_shuffle=bool(self.shuffle),
            dg_patience=self.patience,
            dg_weight_decay=float(self.weight_decay),
            dg_ur_weight=float(self.ur_weight),
            dg_ur_target=self.ur_target,
            zeta_lambda=resolved_zeta_lambda,
            zeta_theta=resolved_zeta_theta,
            use_lse=bool(self.use_lse),
            finetune_epochs=int(self.finetune_epochs),
            finetune_learning_rate=float(self.learning_rate),
            finetune_batch_size=self._effective_batch_size,
            finetune_shuffle=bool(self.shuffle),
            finetune_patience=self.patience,
            finetune_restore_best=bool(self.restore_best),
            finetune_weight_decay=float(self.weight_decay),
            finetune_ur_weight=float(self.ur_weight),
            finetune_ur_target=self.ur_target,
            verbose=self.verbose,
            eval_metrics_every=self.eval_metrics_every,
            scheduler_class=self.scheduler_class,
            scheduler_params=self.scheduler_params,
            optimizer_type=self.optimizer_type,
            structural_pruning=bool(self.structural_pruning),
            finetune_freeze_antecedents=bool(self.freeze_antecedents_finetune),
        )

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
        metrics: list[str] | None = None,
    ) -> DGALETSKRegressor:
        """Train the DG-ALETSK regressor.

        Validation data should be supplied using ``x_val`` and ``y_val``
        when available.
        """
        return cast(DGALETSKRegressor, super().fit(x, y, x_val=x_val, y_val=y_val, metrics=metrics))

    def __sklearn_tags__(self) -> Any:
        """Mark as poor_score: DG-ALETSK is designed for high-dimensional data."""
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags
