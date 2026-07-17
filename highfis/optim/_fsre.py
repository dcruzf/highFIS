"""Three-phase FSRE trainer for FSRE-ADATSK models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from ..models import BaseTSK
from ..models._common import _threshold_from_zeta
from ._base import BaseTrainer
from ._protocols import FSREModelProtocol

_DEFAULT_ZETA_LAMBDA: float = 0.5
_DEFAULT_ZETA_THETA: float = 0.3


class FSRETrainer(BaseTrainer):
    """Three-phase trainer for FSRE-ADATSK classifier and regressor models.

    Implements FSRE Algorithm, which consists of three
    sequential phases:

    1. **FS phase** — Train with CoCo-FRB and feature gates M(λ_d) active
       (paper eq. 21).  After training, features with gate activation
       M(λ_d) > τ_λ are retained.
    2. **RE phase** — Expand to En-FRB and train with rule gates M(θ_r) active
       (paper eq. 22).  After training, rules with gate activation
       M(θ_r) > τ_θ are retained.  For classifiers, at least *n_classes*
       rules are kept.
    3. **Fine-tune phase** — Train the pruned model without gates (paper eq. 5).

    Thresholds are computed directly from scalar zeta coefficients.

    Example:
        ```python
        from highfis import FSREADATSKClassifier
        from highfis.optim import FSRETrainer

        trainer = FSRETrainer(
            fs_epochs=10,
            re_epochs=10,
            finetune_epochs=100,
            zeta_lambda=0.5,
            zeta_theta=0.3,
        )
        clf = FSREADATSKClassifier(trainer=trainer)
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        # ── Phase 1: FS ──────────────────────────────────────────────────
        fs_epochs: int = 10,
        fs_learning_rate: float = 1e-2,
        fs_batch_size: int | None = 512,
        fs_shuffle: bool = True,
        fs_patience: int | None = 20,
        fs_weight_decay: float = 1e-8,
        fs_ur_weight: float = 0.0,
        fs_ur_target: float | None = None,
        # ── Phase 2: RE ──────────────────────────────────────────────────
        re_epochs: int = 10,
        re_learning_rate: float = 1e-2,
        re_batch_size: int | None = 512,
        re_shuffle: bool = True,
        re_patience: int | None = 20,
        re_weight_decay: float = 1e-8,
        re_ur_weight: float = 0.0,
        re_ur_target: float | None = None,
        # ── Phase 3: Fine-tune ────────────────────────────────────────────
        finetune_epochs: int = 100,
        finetune_learning_rate: float = 1e-2,
        finetune_batch_size: int | None = 512,
        finetune_shuffle: bool = True,
        finetune_patience: int | None = 20,
        finetune_restore_best: bool = True,
        finetune_weight_decay: float = 1e-8,
        finetune_ur_weight: float = 0.0,
        finetune_ur_target: float | None = None,
        # ── Thresholds ────────────────────────────────────────────────────
        zeta_lambda: float = _DEFAULT_ZETA_LAMBDA,
        zeta_theta: float = _DEFAULT_ZETA_THETA,
        # ── Pruning ───────────────────────────────────────────────────────
        structural_pruning: bool = True,
        # ── Shared ────────────────────────────────────────────────────────
        verbose: bool | int = False,
        loss: Callable[..., Any] | None = None,
        eval_metrics_every: int = 1,
        scheduler_class: type[Any] | None = None,
        scheduler_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialise an FSRE trainer.

        Args:
            fs_epochs: Epochs for the FS phase (phase 1).  Default ``10``.
            fs_learning_rate: Adam learning rate for the FS phase.
            fs_batch_size: Mini-batch size for the FS phase.
            fs_shuffle: Reshuffle samples each epoch in the FS phase.
            fs_patience: Early-stopping patience for the FS phase.
            fs_weight_decay: L2 weight-decay for the FS phase.
            fs_ur_weight: Uncertainty regularisation weight for the FS phase.
            fs_ur_target: Uncertainty regularisation target for the FS phase.
            re_epochs: Epochs for the RE phase (phase 2).  Default ``10``.
            re_learning_rate: Adam learning rate for the RE phase.
            re_batch_size: Mini-batch size for the RE phase.
            re_shuffle: Reshuffle samples each epoch in the RE phase.
            re_patience: Early-stopping patience for the RE phase.
            re_weight_decay: L2 weight-decay for the RE phase.
            re_ur_weight: Uncertainty regularisation weight for the RE phase.
            re_ur_target: Uncertainty regularisation target for the RE phase.
            finetune_epochs: Epochs for the fine-tune phase (phase 3).
                Default ``100``.
            finetune_learning_rate: Adam learning rate for fine-tuning.
            finetune_batch_size: Mini-batch size for fine-tuning.
            finetune_shuffle: Reshuffle samples each epoch during fine-tuning.
            finetune_patience: Early-stopping patience for fine-tuning.
            finetune_restore_best: Restore best validation weights after
                fine-tuning.
            finetune_weight_decay: L2 weight-decay for fine-tuning.
            finetune_ur_weight: Uncertainty regularisation weight for
                fine-tuning.
            finetune_ur_target: Uncertainty regularisation target for
                fine-tuning.
            zeta_lambda: Coefficient to compute the feature-selection
                threshold τ_λ (paper eq. 28).  Larger values retain more
                features; ``0.5`` is recommended for low-dimensional data,
                ``0.4`` for high-dimensional data.
            zeta_theta: Coefficient to compute the rule-extraction threshold
                τ_θ (paper eq. 29).  ``0.3`` is recommended for
                low-dimensional data, ``0.5`` for high-dimensional data.
            structural_pruning: If ``True`` (default), hard-prune the model
                architecture after each threshold step.  If ``False``, only
                the gate values are modified but the model structure is
                unchanged.
            verbose: Verbosity level forwarded to all three phases.
            loss: Custom loss function ``f(output, target) -> scalar``.
                ``None`` uses the model's built-in criterion.
            eval_metrics_every: Evaluate training metrics every ``n`` epochs in each
                gradient phase; ``0`` skips them. See
                :class:`~highfis.optim.GradientTrainer`.
            scheduler_class: Learning-rate scheduler *class*, applied to every gradient
                phase. Each phase builds its own optimiser, so each also gets its own
                scheduler instance -- one shared instance could not span them.
            scheduler_params: Keyword arguments for ``scheduler_class``.
        """
        self.fs_epochs = fs_epochs
        self.fs_learning_rate = fs_learning_rate
        self.fs_batch_size = fs_batch_size
        self.fs_shuffle = fs_shuffle
        self.fs_patience = fs_patience
        self.fs_weight_decay = fs_weight_decay
        self.fs_ur_weight = fs_ur_weight
        self.fs_ur_target = fs_ur_target

        self.re_epochs = re_epochs
        self.re_learning_rate = re_learning_rate
        self.re_batch_size = re_batch_size
        self.re_shuffle = re_shuffle
        self.re_patience = re_patience
        self.re_weight_decay = re_weight_decay
        self.re_ur_weight = re_ur_weight
        self.re_ur_target = re_ur_target

        self.finetune_epochs = finetune_epochs
        self.finetune_learning_rate = finetune_learning_rate
        self.finetune_batch_size = finetune_batch_size
        self.finetune_shuffle = finetune_shuffle
        self.finetune_patience = finetune_patience
        self.finetune_restore_best = finetune_restore_best
        self.finetune_weight_decay = finetune_weight_decay
        self.finetune_ur_weight = finetune_ur_weight
        self.finetune_ur_target = finetune_ur_target

        self.zeta_lambda = zeta_lambda
        self.zeta_theta = zeta_theta
        self.structural_pruning = structural_pruning
        self.verbose = verbose
        self.loss = loss
        self.eval_metrics_every = eval_metrics_every
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params

    def fit(
        self,
        model: BaseTSK,
        x: Tensor,
        y: Tensor,
        *,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute the three-phase FSRE training procedure (Algorithm 1).

        Args:
            model: An FSRE-ADATSK model instance, such as
                FSREADATSKClassifierModel or FSREADATSKRegressorModel.
            x: Training inputs of shape ``(N, D)``.
            y: Training targets.
            x_val: Validation inputs (used for early stopping).
                When ``None``, no external validation is performed.
            y_val: Validation targets.
            metrics: Optional list of metric names to evaluate.

        Returns:
            Dictionary with keys:

            - ``"fs"`` — history dict from phase 1 (FS phase).
            - ``"re"`` — history dict from phase 2 (RE phase).
            - ``"finetune"`` — history dict from phase 3 (fine-tune phase).
            - ``"surviving_feature_indices"`` — list of retained feature indices
                relative to the input ``x`` columns.
            - ``"surviving_rule_indices"`` — list of retained rule indices
                relative to the En-FRB rule count after phase 2.
            - ``"tau_lambda"`` — applied feature-selection threshold.
            - ``"tau_theta"`` — applied rule-extraction threshold.
        """
        from ._gradient import GradientTrainer

        if not isinstance(model, FSREModelProtocol):
            raise TypeError("model must implement FSREModelProtocol")

        # ── Phase 1: Feature Selection ────────────────────────────────────
        # Set consequent to FS mode (only feature gates M(λ_d) active)
        model.set_consequent_mode("fs")
        fs_trainer = GradientTrainer(
            epochs=int(self.fs_epochs),
            learning_rate=float(self.fs_learning_rate),
            loss=self.loss,
            eval_metrics_every=self.eval_metrics_every,
            scheduler_class=self.scheduler_class,
            scheduler_params=self.scheduler_params,
            batch_size=self.fs_batch_size,
            shuffle=bool(self.fs_shuffle),
            ur_weight=float(self.fs_ur_weight),
            ur_target=self.fs_ur_target,
            verbose=self.verbose,
            patience=self.fs_patience,
            weight_decay=float(self.fs_weight_decay),
        )
        fs_history: dict[str, Any] = fs_trainer.fit(model, x, y, x_val=x_val, y_val=y_val, metrics=metrics)

        # ── Feature threshold & selection (paper eq. 28) ──────────────────
        feat_gates: Tensor = model.get_feature_gate_values()
        tau_lambda: float = _threshold_from_zeta(feat_gates, float(self.zeta_lambda))
        sf: list[int] = [i for i, v in enumerate(feat_gates.tolist()) if v > tau_lambda]
        if not sf:
            # Edge case: keep the single most-activated feature.
            sf = [int(feat_gates.argmax().item())]

        if self.structural_pruning:
            model.prune_to_features(sf)
            x_fs: Tensor = x[:, sf]
            x_val_fs: Tensor | None = x_val[:, sf] if x_val is not None else None
        else:
            x_fs, x_val_fs = x, x_val

        # ── Phase 2: Rule Extraction ──────────────────────────────────────
        # Expand to En-FRB (rebuilds rule_layer and consequent_layer with mode="re")
        model.expand_to_en_frb()
        re_trainer = GradientTrainer(
            epochs=int(self.re_epochs),
            learning_rate=float(self.re_learning_rate),
            loss=self.loss,
            eval_metrics_every=self.eval_metrics_every,
            scheduler_class=self.scheduler_class,
            scheduler_params=self.scheduler_params,
            batch_size=self.re_batch_size,
            shuffle=bool(self.re_shuffle),
            ur_weight=float(self.re_ur_weight),
            ur_target=self.re_ur_target,
            verbose=self.verbose,
            patience=self.re_patience,
            weight_decay=float(self.re_weight_decay),
        )
        re_history: dict[str, Any] = re_trainer.fit(model, x_fs, y, x_val=x_val_fs, y_val=y_val, metrics=metrics)

        # ── Rule threshold & selection (paper eq. 29) ─────────────────────
        rule_gates: Tensor = model.get_rule_gate_values()
        tau_theta: float = _threshold_from_zeta(rule_gates, float(self.zeta_theta))
        sr: list[int] = [r for r, v in enumerate(rule_gates.tolist()) if v > tau_theta]

        # Enforce lower bound: for classifiers ≥ n_classes rules (paper §III-C).
        from ..models._common import BaseTSKClassifierModel

        min_rules: int = 1
        if isinstance(model, BaseTSKClassifierModel):
            min_rules = model.n_classes
        if len(sr) < min_rules:
            top_indices: list[int] = torch.topk(rule_gates, min_rules).indices.tolist()
            sr = sorted(top_indices)

        if self.structural_pruning:
            model.prune_to_rules(sr)

        # Set consequent to finetune mode (no gates — plain TSK consequent)
        model.set_consequent_mode("finetune")
        ft_trainer = GradientTrainer(
            epochs=int(self.finetune_epochs),
            learning_rate=float(self.finetune_learning_rate),
            loss=self.loss,
            eval_metrics_every=self.eval_metrics_every,
            scheduler_class=self.scheduler_class,
            scheduler_params=self.scheduler_params,
            batch_size=self.finetune_batch_size,
            shuffle=bool(self.finetune_shuffle),
            ur_weight=float(self.finetune_ur_weight),
            ur_target=self.finetune_ur_target,
            verbose=self.verbose,
            patience=self.finetune_patience,
            restore_best=bool(self.finetune_restore_best),
            weight_decay=float(self.finetune_weight_decay),
        )
        finetune_history: dict[str, Any] = ft_trainer.fit(model, x_fs, y, x_val=x_val_fs, y_val=y_val, metrics=metrics)

        return {
            "fs": fs_history,
            "re": re_history,
            "finetune": finetune_history,
            "surviving_feature_indices": sf,
            "surviving_rule_indices": sr,
            "tau_lambda": tau_lambda,
            "tau_theta": tau_theta,
        }
