"""Three-phase DG trainer for DG-TSK and DG-ALETSK models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from torch import Tensor

from ..base import BaseTSK
from ._base import BaseTrainer
from ._protocols import DGModelProtocol

_DEFAULT_ZETA: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]


class DGTrainer(BaseTrainer):
    """Three-phase trainer for DG-TSK and DG-ALETSK estimators.

    Implements the data-guided (DG) training procedure described in Xue et
    al. (2023), which consists of three sequential phases:

    1. **DG phase** — Train gate parameters (λ, θ) and zero-order consequent
       parameters.  For DG-TSK the antecedent MF parameters are **frozen**
       (P-FRB, paper §III-A).  For DG-ALETSK the antecedents are updated
       together with the gates (CoCo-FRB, paper eq. 22).
    2. **Threshold search** — Grid-search over ``(zeta_lambda, zeta_theta)``
       pairs to find the pruning thresholds that maximise held-out
       performance, optionally refitting first-order consequents via LSE.
    3. **Fine-tune phase** — Convert the model to first-order consequents
       and retrain with antecedent MFs and feature gates (λ) **frozen**.

    Each phase is delegated to the corresponding method on the model:
    :meth:`~highfis.models.DGTSKClassifierModel.fit_dg_phase`,
    :meth:`~highfis.models.DGTSKClassifierModel.search_thresholds`, and
    :meth:`~highfis.models.DGTSKClassifierModel.fit_finetune`.

    References:
        * Xue et al., *Fuzzy Sets and Systems*, 2023 (DG-TSK).
          https://doi.org/10.1016/j.fss.2023.108627
        * Xue et al., *IEEE Trans. Fuzzy Systems*, 2023 (DG-ALETSK).
          https://doi.org/10.1109/TFUZZ.2023.3270445

    Example::

        from highfis import DGTSKClassifier
        from highfis.optim import DGTrainer

        trainer = DGTrainer(dg_epochs=20, finetune_epochs=100, use_lse=True)
        clf = DGTSKClassifier(trainer=trainer)
        clf.fit(X_train, y_train, x_val=X_val, y_val=y_val)
    """

    def __init__(
        self,
        *,
        # ── Phase 1: DG ──────────────────────────────────────────────────
        dg_epochs: int = 10,
        dg_learning_rate: float = 1e-2,
        dg_batch_size: int | None = 512,
        dg_shuffle: bool = True,
        dg_patience: int | None = 20,
        dg_weight_decay: float = 1e-8,
        dg_ur_weight: float = 0.0,
        dg_ur_target: float | None = None,
        # ── Phase 2: Threshold search ─────────────────────────────────────
        zeta_lambda: list[float] | None = None,
        zeta_theta: list[float] | None = None,
        use_lse: bool = True,
        # ── Phase 3: Fine-tune ────────────────────────────────────────────
        finetune_epochs: int = 200,
        finetune_learning_rate: float = 1e-2,
        finetune_batch_size: int | None = 512,
        finetune_shuffle: bool = True,
        finetune_patience: int | None = 20,
        finetune_restore_best: bool = True,
        finetune_weight_decay: float = 1e-8,
        finetune_ur_weight: float = 0.0,
        finetune_ur_target: float | None = None,
        # ── Shared ────────────────────────────────────────────────────────
        verbose: bool | int = False,
        loss: Callable[..., Any] | None = None,
    ) -> None:
        """Initialise a DG trainer.

        Args:
            dg_epochs: Epochs for the DG phase (phase 1).
            dg_learning_rate: Adam learning rate for the DG phase.
            dg_batch_size: Mini-batch size for the DG phase.
            dg_shuffle: Reshuffle samples each epoch in the DG phase.
            dg_patience: Early-stopping patience for the DG phase.
            dg_weight_decay: L2 weight-decay for the DG phase.
            dg_ur_weight: Uncertainty regularisation weight for the DG phase.
            dg_ur_target: Uncertainty regularisation target for the DG phase.
            zeta_lambda: Grid of λ-threshold candidates for pruning.  If
                ``None``, uses ``[0.0, 0.25, 0.5, 0.75, 1.0]``.
            zeta_theta: Grid of θ-threshold candidates.  Same default.
            use_lse: Refit first-order consequents via LSE during threshold
                search.  Recommended (default ``True``).
            finetune_epochs: Epochs for the fine-tune phase (phase 3).
            finetune_learning_rate: Adam learning rate for fine-tuning.
            finetune_batch_size: Mini-batch size for fine-tuning.
            finetune_shuffle: Reshuffle samples each epoch during fine-tuning.
            finetune_patience: Early-stopping patience for fine-tuning.
            finetune_restore_best: Restore best validation weights after
                fine-tuning.
            finetune_weight_decay: L2 weight-decay for fine-tuning.
            finetune_ur_weight: Uncertainty regularisation weight for fine-tuning.
            finetune_ur_target: Uncertainty regularisation target for fine-tuning.
            verbose: Verbosity level forwarded to all three phases.
            loss: Custom loss function ``f(output, target) -> scalar``.
                ``None`` uses the model's built-in criterion.
        """
        self.dg_epochs = dg_epochs
        self.dg_learning_rate = dg_learning_rate
        self.dg_batch_size = dg_batch_size
        self.dg_shuffle = dg_shuffle
        self.dg_patience = dg_patience
        self.dg_weight_decay = dg_weight_decay
        self.dg_ur_weight = dg_ur_weight
        self.dg_ur_target = dg_ur_target
        self.zeta_lambda = zeta_lambda
        self.zeta_theta = zeta_theta
        self.use_lse = use_lse
        self.finetune_epochs = finetune_epochs
        self.finetune_learning_rate = finetune_learning_rate
        self.finetune_batch_size = finetune_batch_size
        self.finetune_shuffle = finetune_shuffle
        self.finetune_patience = finetune_patience
        self.finetune_restore_best = finetune_restore_best
        self.finetune_weight_decay = finetune_weight_decay
        self.finetune_ur_weight = finetune_ur_weight
        self.finetune_ur_target = finetune_ur_target
        self.verbose = verbose
        self.loss = loss

    def fit(
        self,
        model: BaseTSK,
        x: Tensor,
        y: Tensor,
        *,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
    ) -> dict[str, Any]:
        """Execute the three-phase DG training procedure.

        Args:
            model: A DG-TSK or DG-ALETSK model instance
                (:class:`~highfis.models.DGTSKClassifierModel`,
                :class:`~highfis.models.DGTSKRegressorModel`,
                :class:`~highfis.models.DGALETSKClassifierModel`, or
                :class:`~highfis.models.DGALETSKRegressorModel`).
            x: Training inputs of shape ``(N, D)``.
            y: Training targets.
            x_val: Validation inputs (used for threshold-search scoring).
                When ``None``, training data is used for threshold selection.
            y_val: Validation targets.

        Returns:
            Dictionary with keys:

            - ``"dg"`` — history dict from phase 1 (DG phase).
            - ``"threshold"`` — result dict from
              :meth:`~highfis.models.DGTSKClassifierModel.search_thresholds`.
            - ``"finetune"`` — history dict from phase 3 (fine-tune phase).
        """
        zeta_lambda = self.zeta_lambda if self.zeta_lambda is not None else _DEFAULT_ZETA
        zeta_theta = self.zeta_theta if self.zeta_theta is not None else _DEFAULT_ZETA

        dg_model = cast(DGModelProtocol, model)

        # ── Phase 1: DG training ──────────────────────────────────────────
        dg_history: dict[str, Any] = dg_model.fit_dg_phase(
            x,
            y,
            epochs=int(self.dg_epochs),
            learning_rate=float(self.dg_learning_rate),
            criterion=self.loss,
            batch_size=self.dg_batch_size,
            shuffle=bool(self.dg_shuffle),
            ur_weight=float(self.dg_ur_weight),
            ur_target=self.dg_ur_target,
            verbose=self.verbose,
            x_val=x_val,
            y_val=y_val,
            patience=self.dg_patience,
            weight_decay=float(self.dg_weight_decay),
        )

        # ── Phase 2: Threshold search + pruning ───────────────────────────
        # Fall back to training data for threshold scoring when no hold-out
        # set is provided (paper practice).
        x_eval = x_val if x_val is not None else x
        y_eval = y_val if y_val is not None else y

        threshold_result: dict[str, Any] = dg_model.search_thresholds(
            x,
            y,
            zeta_lambda=zeta_lambda,
            zeta_theta=zeta_theta,
            x_val=x_eval,
            y_val=y_eval,
            use_lse=bool(self.use_lse),
            inplace=True,
            verbose=bool(self.verbose),
        )

        # ── Phase 3: Fine-tune ────────────────────────────────────────────
        finetune_history: dict[str, Any] = dg_model.fit_finetune(
            x,
            y,
            epochs=int(self.finetune_epochs),
            learning_rate=float(self.finetune_learning_rate),
            criterion=self.loss,
            batch_size=self.finetune_batch_size,
            shuffle=bool(self.finetune_shuffle),
            ur_weight=float(self.finetune_ur_weight),
            ur_target=self.finetune_ur_target,
            verbose=self.verbose,
            x_val=x_val,
            y_val=y_val,
            patience=self.finetune_patience,
            restore_best=bool(self.finetune_restore_best),
            weight_decay=float(self.finetune_weight_decay),
        )

        return {
            "dg": dg_history,
            "threshold": threshold_result,
            "finetune": finetune_history,
        }
