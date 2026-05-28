"""Sklearn-compatible estimators for DG-ALETSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseTSK
from ..memberships import MembershipFunction
from ..models import (
    DGALETSKClassifierModel,
    DGALETSKRegressorModel,
)
from ..optim._base import BaseTrainer
from ..optim._dg import DGTrainer
from ._base import InputConfig
from ._fsre import (
    FSREADATSKClassifier,
    FSREADATSKRegressor,
)


class DGALETSKClassifier(FSREADATSKClassifier):
    """DG-ALETSK classifier with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-ADATSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.  Training follows the three-phase DG protocol:

    1. **DG phase** — Train gate parameters, antecedent MFs, and zero-order
       consequents (CoCo-FRB; antecedents are **not** frozen unlike DG-TSK).
    2. **Threshold search** — Grid-search for the pruning thresholds
       ``(zeta_lambda, zeta_theta)`` that maximise held-out accuracy.
    3. **Fine-tune phase** — Train first-order consequents with antecedent
       MFs and feature gates frozen.

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
        optimizer_type: str = "sgd",
        structural_pruning: bool = True,
        freeze_antecedents_finetune: bool = True,
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
            dg_epochs: Maximum epochs for phase 1 (DG training).
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
            optimizer_type: Optimizer type.  ``"sgd"`` (default, paper) or
                ``"adamw"``.
            structural_pruning: If ``True`` (default), apply hard structural
                pruning after threshold search.
            freeze_antecedents_finetune: If ``True`` (default), freeze MF
                parameters and feature gates during fine-tuning.
        """
        self.dg_epochs = int(dg_epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.zeta_lambda = zeta_lambda
        self.zeta_theta = zeta_theta
        self.use_lse = bool(use_lse)
        self.optimizer_type = str(optimizer_type)
        self.structural_pruning = bool(structural_pruning)
        self.freeze_antecedents_finetune = bool(freeze_antecedents_finetune)
        super().__init__(
            lambda_init=lambda_init,
            use_en_frb=use_en_frb,
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=dg_epochs,
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
        )

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create DGALETSKClassifierModel."""
        return DGALETSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
            optimizer_type=self.optimizer_type,
        )

    def _get_trainer(self) -> DGTrainer:
        """Return a :class:`~highfis.optim.DGTrainer` built from this estimator's params."""
        return DGTrainer(
            dg_epochs=int(self.dg_epochs),
            dg_learning_rate=float(self.learning_rate),
            dg_batch_size=self.batch_size,
            dg_shuffle=bool(self.shuffle),
            dg_patience=self.patience,
            dg_weight_decay=float(self.weight_decay),
            dg_ur_weight=float(self.ur_weight),
            dg_ur_target=self.ur_target,
            zeta_lambda=self.zeta_lambda,
            zeta_theta=self.zeta_theta,
            use_lse=bool(self.use_lse),
            finetune_epochs=int(self.finetune_epochs),
            finetune_learning_rate=float(self.learning_rate),
            finetune_batch_size=self.batch_size,
            finetune_shuffle=bool(self.shuffle),
            finetune_patience=self.patience,
            finetune_restore_best=bool(self.restore_best),
            finetune_weight_decay=float(self.weight_decay),
            finetune_ur_weight=float(self.ur_weight),
            finetune_ur_target=self.ur_target,
            verbose=self.verbose,
            optimizer_type=self.optimizer_type,
            structural_pruning=bool(self.structural_pruning),
            finetune_freeze_antecedents=bool(self.freeze_antecedents_finetune),
        )


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
        optimizer_type: str = "sgd",
        structural_pruning: bool = True,
        freeze_antecedents_finetune: bool = True,
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
            dg_epochs: Maximum epochs for phase 1 (DG training).
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
            optimizer_type: Optimizer type.  ``"sgd"`` (default, paper) or
                ``"adamw"``.
            structural_pruning: If ``True`` (default), apply hard structural
                pruning after threshold search.
            freeze_antecedents_finetune: If ``True`` (default), freeze MF
                parameters and feature gates during fine-tuning.
        """
        self.dg_epochs = int(dg_epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.zeta_lambda = zeta_lambda
        self.zeta_theta = zeta_theta
        self.use_lse = bool(use_lse)
        self.optimizer_type = str(optimizer_type)
        self.structural_pruning = bool(structural_pruning)
        self.freeze_antecedents_finetune = bool(freeze_antecedents_finetune)
        super().__init__(
            lambda_init=lambda_init,
            use_en_frb=use_en_frb,
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=dg_epochs,
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
        )

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create DGALETSKRegressorModel."""
        return DGALETSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
            optimizer_type=self.optimizer_type,
        )

    def _get_trainer(self) -> DGTrainer:
        """Return a :class:`~highfis.optim.DGTrainer` built from this estimator's params."""
        return DGTrainer(
            dg_epochs=int(self.dg_epochs),
            dg_learning_rate=float(self.learning_rate),
            dg_batch_size=self.batch_size,
            dg_shuffle=bool(self.shuffle),
            dg_patience=self.patience,
            dg_weight_decay=float(self.weight_decay),
            dg_ur_weight=float(self.ur_weight),
            dg_ur_target=self.ur_target,
            zeta_lambda=self.zeta_lambda,
            zeta_theta=self.zeta_theta,
            use_lse=bool(self.use_lse),
            finetune_epochs=int(self.finetune_epochs),
            finetune_learning_rate=float(self.learning_rate),
            finetune_batch_size=self.batch_size,
            finetune_shuffle=bool(self.shuffle),
            finetune_patience=self.patience,
            finetune_restore_best=bool(self.restore_best),
            finetune_weight_decay=float(self.weight_decay),
            finetune_ur_weight=float(self.ur_weight),
            finetune_ur_target=self.ur_target,
            verbose=self.verbose,
            optimizer_type=self.optimizer_type,
            structural_pruning=bool(self.structural_pruning),
            finetune_freeze_antecedents=bool(self.freeze_antecedents_finetune),
        )
