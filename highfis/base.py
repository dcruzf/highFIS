"""Base TSK model that factors the common antecedent-defuzzification pipeline.

This module defines `BaseTSK`, the abstract foundation for all TSK fuzzy
models in highFIS. It factors out the shared antecedent pipeline,
defuzzifier, and training loop so concrete subclasses can focus on
task-specific consequent layers and loss criteria.

The forward pipeline executes four sequential steps:

1. `highfis.layers.MembershipLayer` — evaluates membership functions for each
   input feature.
2. `highfis.layers.RuleLayer` — computes rule firing strengths via a
   configurable rule base and T-norm.
3. **Defuzzifier** — normalizes firing strengths to probability-like weights
   (default: `highfis.defuzzifiers.SoftmaxLogDefuzzifier`).
4. **ConsequentLayer** — produces the final output from the inputs and the
   normalized rule weights.

Concrete subclasses must implement:

- `BaseTSK._build_consequent_layer` — return the task-specific consequent
  module.
- `BaseTSK._default_criterion` — return the default loss function.

Optional overridable hooks:

- `BaseTSK._compute_loss` — customize target preparation or loss composition.
- `BaseTSK._evaluate_validation` — customize the validation metric used for
  early stopping.
"""

from __future__ import annotations

import copy
import logging
import sys
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn
from tqdm.auto import trange

from .defuzzifiers import SoftmaxLogDefuzzifier
from .layers import MembershipLayer, RuleLayer
from .memberships import MembershipFunction
from .protocols import Defuzzifier
from .t_norms import TNormFn

logger: logging.Logger = logging.getLogger(__name__)


def _uniform_regularization_loss(normalized_weights: Tensor, target: float | None = None) -> Tensor:
    """Penalize deviation from a uniform average rule activation distribution."""
    n_rules = normalized_weights.shape[1]
    target_value = (1.0 / float(n_rules)) if target is None else float(target)
    target_tensor = normalized_weights.new_full((n_rules,), target_value)
    avg_activation = normalized_weights.mean(dim=0)
    return torch.sum((avg_activation - target_tensor) ** 2)


def _iter_minibatch_indices(n_samples: int, batch_size: int | None, shuffle: bool) -> list[Tensor]:
    """Create mini-batch index tensors for one epoch."""
    if batch_size is None:
        return [torch.arange(n_samples)]
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0 when provided")
    if batch_size >= n_samples:
        return [torch.arange(n_samples)]

    order = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    batches: list[Tensor] = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batches.append(order[start:end])
    return batches


class BaseTSK(nn.Module):
    """Abstract base for TSK fuzzy models.

    Subclasses must implement :meth:`_build_consequent_layer` and
    :meth:`_default_criterion`.  Optionally override :meth:`_compute_loss`
    and :meth:`_evaluate_validation` for task-specific logic.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        *,
        rule_base: str = "cartesian",
        t_norm: str | TNormFn = "gmean",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: Defuzzifier | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize the TSK pipeline layers.

        Args:
            input_mfs: Mapping from feature names to sequences of
                :class:`~highfis.memberships.MembershipFunction` objects.
                Must not be empty.
            rule_base: Rule-base construction strategy.  Supported values:
                ``"cartesian"`` (all MF combinations), ``"coco"``
                (same-index compact), ``"en"`` (enhanced FRB), or
                ``"custom"`` (explicit rules via *rules*).
            t_norm: T-norm name or callable.  Common string values: ``"prod"``,
                ``"gmean"``, ``"min"``, ``"dombi"``, ``"yager"``.  A
                callable implementing the T-norm interface may be passed
                directly.
            rules: Explicit rule index sequences.  Required when
                *rule_base* is ``"custom"``.
            defuzzifier: Normalization module applied to raw rule firing
                strengths.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: If ``True``, insert a
                :class:`~torch.nn.BatchNorm1d` layer on the inputs before
                the consequent computation.

        Raises:
            ValueError: If *input_mfs* is empty.
        """
        super().__init__()
        if not input_mfs:
            raise ValueError("input_mfs must not be empty")

        self.input_mfs = input_mfs
        self.input_names = list(input_mfs.keys())
        self.n_inputs = len(self.input_names)
        mf_per_input = [len(input_mfs[name]) for name in self.input_names]

        self.membership_layer = MembershipLayer(input_mfs)
        self.rule_layer = RuleLayer(
            self.input_names,
            mf_per_input,
            rules=rules,
            rule_base=rule_base,
            t_norm=t_norm,
        )
        self.n_rules = self.rule_layer.n_rules
        self.defuzzifier: Defuzzifier = defuzzifier or SoftmaxLogDefuzzifier()
        self.consequent_batch_norm = bool(consequent_batch_norm)
        self.consequent_bn = nn.BatchNorm1d(self.n_inputs) if self.consequent_batch_norm else None
        self.consequent_layer = self._build_consequent_layer()
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(stream_handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_consequent_layer(self) -> nn.Module:
        """Return the task-specific consequent layer."""

    @abstractmethod
    def _default_criterion(self) -> nn.Module:
        """Return the default loss function for this task."""

    # ------------------------------------------------------------------
    # Shared forward pipeline
    # ------------------------------------------------------------------

    def forward_antecedents(self, x: Tensor) -> Tensor:
        """Compute normalized rule strengths from model antecedents."""
        mu = self.membership_layer(x)
        w = self.rule_layer(mu)
        return self.defuzzifier(w)

    def get_mf_params(self) -> dict[str, list[dict[str, Any]]]:
        """Return a serializable description of the model's membership functions."""
        return {
            name: [{"type": type(mf).__name__, **mf.inspect_params()} for mf in cast(Sequence[MembershipFunction], mfs)]
            for name, mfs in self.membership_layer.input_mfs.items()
        }

    def get_rule_table(self) -> list[dict[str, Any]]:
        """Return the rule base as a table of feature-to-MF indices."""
        return [
            dict(
                rule_id=rule_index,
                **dict(zip(self.rule_layer.input_names, rule, strict=False)),
            )
            for rule_index, rule in enumerate(self.rule_layer.rules)
        ]

    def get_consequent_weights(self) -> Tensor | None:
        """Return the consequent layer weights or ``None`` when unavailable."""
        weight = getattr(self.consequent_layer, "weight", None)
        if isinstance(weight, Tensor):
            return weight.detach()
        return None

    def _forward_train(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass returning ``(output, norm_w)`` to avoid double computation."""
        mu = self.membership_layer(x)
        w = self.rule_layer(mu)
        norm_w = self.defuzzifier(w)
        x_cons = self.consequent_bn(x) if self.consequent_bn is not None else x
        output = self.consequent_layer(x_cons, norm_w)
        return output, norm_w

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass through the TSK pipeline."""
        output, _ = self._forward_train(x)
        return output

    # ------------------------------------------------------------------
    # Training helpers (may be overridden)
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        criterion: Callable[[Tensor, Tensor], Tensor],
        output: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute the main task loss.  Override for custom target preparation."""
        return criterion(output, target)

    def _log(
        self,
        message: str,
        *args: Any,
        level: int = logging.INFO,
        verbose: bool | int = False,
        min_level: int = 2,
        **kwargs: Any,
    ) -> None:
        """Log a message when verbose mode is enabled."""
        if self._resolve_verbose(verbose) < min_level:
            return
        self.logger.log(level, message, *args, **kwargs)

    def _resolve_verbose(self, verbose: bool | int = False) -> int:
        """Normalize verbose settings to a numeric verbosity level."""
        if isinstance(verbose, bool):
            return 1 if verbose else 0
        if not isinstance(verbose, int):
            raise TypeError("verbose must be an int in 0..3 or a bool")
        if verbose < 0 or verbose > 3:
            raise ValueError("verbose must be between 0 and 3")
        return verbose

    def _evaluate_validation(
        self,
        criterion: Callable[[Tensor, Tensor], Tensor],
        x_val: Tensor,
        y_val: Tensor,
    ) -> dict[str, float]:
        """Evaluate on validation set.  Return dict with at least ``'metric'``.

        The ``'metric'`` value is used for early-stopping comparison.
        By default it is the negated validation loss (higher is better).
        """
        with torch.no_grad():
            output = self.forward(x_val)
            val_loss = float(self._compute_loss(criterion, output, y_val).item())
        return {"val_loss": val_loss, "metric": -val_loss}

    # ------------------------------------------------------------------
    # Private fit helpers
    # ------------------------------------------------------------------

    def _validate_fit_inputs(
        self,
        x: Tensor,
        y: Tensor,
        x_val: Tensor | None,
        y_val: Tensor | None,
        ur_weight: float,
        ur_target: float | None,
    ) -> bool:
        """Validate all inputs to :meth:`fit` and return ``has_val``."""
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"expected x shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
        if y.ndim != 1:
            raise ValueError("expected y shape (batch,)")
        if ur_weight < 0.0:
            raise ValueError("ur_weight must be >= 0")
        if ur_target is not None and not (0.0 < ur_target <= 1.0):
            raise ValueError("ur_target must be in (0, 1] when provided")

        has_val = x_val is not None and y_val is not None
        if has_val and x_val is not None and y_val is not None:
            if x_val.ndim != 2 or x_val.shape[1] != self.n_inputs:
                raise ValueError(f"expected x_val shape (batch, {self.n_inputs}), got {tuple(x_val.shape)}")
            if y_val.ndim != 1:
                raise ValueError("expected y_val shape (batch,)")
        return has_val

    def _build_optimizer(
        self,
        optimizer: torch.optim.Optimizer | None,
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Return *optimizer* unchanged, or build a default AdamW."""
        if optimizer is not None:
            return optimizer
        ante_params = list(self.membership_layer.parameters())
        rule_params = list(self.rule_layer.parameters())
        cons_params = list(self.consequent_layer.parameters())
        if self.consequent_bn is not None:
            cons_params.extend(self.consequent_bn.parameters())
        return torch.optim.AdamW(
            [
                {"params": ante_params, "weight_decay": 0.0},
                {"params": rule_params, "weight_decay": 0.0},
                {"params": cons_params, "weight_decay": weight_decay},
            ],
            lr=learning_rate,
        )

    def _run_minibatch_epoch(
        self,
        x: Tensor,
        y: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        batch_size: int | None,
        shuffle: bool,
        ur_weight: float,
        ur_target: float | None,
    ) -> tuple[float, float]:
        """Run one full epoch of mini-batch gradient updates.

        Returns:
            ``(mean_train_loss, mean_ur_loss)`` averaged over all batches.
        """
        batch_losses: list[float] = []
        batch_ur_losses: list[float] = []
        for batch_idx in _iter_minibatch_indices(x.shape[0], batch_size=batch_size, shuffle=shuffle):
            x_b = x.index_select(0, batch_idx.to(device=x.device))
            y_b = y.index_select(0, batch_idx.to(device=y.device))

            optimizer.zero_grad(set_to_none=True)
            output, norm_w = self._forward_train(x_b)
            main_loss = self._compute_loss(criterion, output, y_b)

            ur_loss = _uniform_regularization_loss(norm_w, target=ur_target)
            total_loss = main_loss + float(ur_weight) * ur_loss
            total_loss.backward()
            optimizer.step()

            batch_losses.append(float(total_loss.detach().item()))
            batch_ur_losses.append(float(ur_loss.detach().item()))

        train_loss = float(sum(batch_losses) / max(len(batch_losses), 1))
        ur_loss_avg = float(sum(batch_ur_losses) / max(len(batch_ur_losses), 1))
        return train_loss, ur_loss_avg

    def _log_epoch_with_val(
        self,
        epoch: int,
        epochs: int,
        train_loss: float,
        val_info: dict[str, Any],
        verbose_level: int,
        pbar: Any,
    ) -> None:
        """Emit epoch-level progress when a validation set is active."""
        if verbose_level == 1:
            if pbar is None:  # pragma: no cover
                raise RuntimeError("progress bar unavailable for verbose level 1")
            postfix = [
                f"train={train_loss:.4f}",
                f"val={val_info.get('val_loss', 0.0):.4f}",
            ]
            pbar.set_postfix_str(" ".join(postfix))
        if verbose_level >= 2 and (verbose_level == 3 or ((epoch + 1) % max(epochs // 10, 1) == 0 or epoch == 0)):
            log_parts = [
                f"epoch={epoch + 1}/{epochs}",
                f"train_loss={train_loss:.6f}",
            ]
            for k, v in val_info.items():
                if k != "metric":
                    log_parts.append(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")
            self._log(" ".join(log_parts), verbose=verbose_level)

    def _log_epoch_no_val(
        self,
        epoch: int,
        epochs: int,
        train_loss: float,
        verbose_level: int,
        pbar: Any,
    ) -> None:
        """Emit epoch-level progress when no validation set is provided."""
        if verbose_level == 1:
            if pbar is None:  # pragma: no cover
                raise RuntimeError("progress bar unavailable for verbose level 1")
            pbar.set_postfix_str(f"loss={train_loss:.4f}")
        if verbose_level >= 2 and (verbose_level == 3 or ((epoch + 1) % max(epochs // 10, 1) == 0 or epoch == 0)):
            self._log(
                "epoch=%s/%s loss=%.6f",
                epoch + 1,
                epochs,
                train_loss,
                verbose=verbose_level,
            )

    # ------------------------------------------------------------------
    # Unified training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        criterion: Callable[[Tensor, Tensor], Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        verbose: bool | int = False,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
    ) -> dict[str, Any]:
        """Train the model with optional early stopping.

        When *x_val* and *y_val* are provided the model evaluates a
        task-specific metric (via :meth:`_evaluate_validation`) after every
        epoch and applies early stopping when the metric has not improved for
        *patience* consecutive epochs.
        By default the best model weights from validation are restored when
        ``restore_best=True``.

        Args:
            x: Training features of shape ``(N, n_inputs)``.
            y: Training targets of shape ``(N,)``.
            epochs: Maximum number of training epochs.
            learning_rate: Learning rate for the default AdamW optimizer.
            criterion: Optional loss function.  Defaults to
                :meth:`_default_criterion`.
            optimizer: Optional pre-built optimizer.  When ``None``, AdamW
                is constructed with separate parameter groups for antecedent
                (no weight decay) and consequent (*weight_decay*) layers.
            batch_size: Mini-batch size.  ``None`` uses the full dataset.
            shuffle: If ``True``, reshuffle sample indices each epoch.
            ur_weight: Non-negative weight for the uniform rule
                regularization term.  ``0.0`` disables it.
            ur_target: Target uniform activation for UR.  Must be in
                ``(0, 1]`` when provided.  ``None`` defaults to
                ``1 / n_rules``.
            verbose: Verbosity level. ``0`` = quiet, ``1`` = progress bar,
                ``2`` = per-epoch summary logging, ``3`` = per-epoch detailed
                logging. ``True`` is accepted as an alias for ``1``.
            x_val: Optional validation features of shape
                ``(M, n_inputs)``.
            y_val: Optional validation targets of shape ``(M,)``.
            patience: Number of consecutive epochs without improvement
                before early stopping.  Set to ``None`` to disable early
                stopping.  Only active when *x_val* and *y_val* are given.
            restore_best: If ``True`` (default), restore the model weights
                from the best validation epoch when early stopping is used.
            weight_decay: L2 weight decay applied to consequent parameters
                by the default AdamW optimizer.

        Returns:
            A dictionary with keys ``"train"``, ``"ur"``, and ``"val"``
            containing per-epoch loss lists.

        Raises:
            ValueError: If shapes of *x*, *y*, *x_val*, or *y_val* are
                incompatible, or if *ur_weight* < 0 or *ur_target* is
                outside ``(0, 1]``.
        """
        has_val = self._validate_fit_inputs(x, y, x_val, y_val, ur_weight, ur_target)
        train_criterion = criterion or self._default_criterion()
        train_optimizer = self._build_optimizer(optimizer, learning_rate, weight_decay)

        history: dict[str, Any] = {"train": [], "ur": [], "val": []}
        best_metric = float("-inf")
        epochs_no_improve = 0
        best_state: dict[str, Any] | None = None
        verbose_level = self._resolve_verbose(verbose)

        self.train()
        pbar = None
        if verbose_level == 1:
            pbar = trange(epochs, desc="Training", leave=False)
            epoch_iterator = pbar
        else:
            epoch_iterator = range(epochs)

        stopped_epoch = 0
        for epoch in epoch_iterator:
            stopped_epoch = epoch + 1
            epoch_train_loss, epoch_ur_loss = self._run_minibatch_epoch(
                x, y, train_criterion, train_optimizer, batch_size, shuffle, ur_weight, ur_target
            )
            history["train"].append(epoch_train_loss)
            history["ur"].append(epoch_ur_loss)

            if has_val and x_val is not None and y_val is not None:
                self.eval()
                val_info = self._evaluate_validation(train_criterion, x_val, y_val)
                history["val"].append(val_info.get("val_loss", 0.0))
                for k, v in val_info.items():
                    if k not in ("val_loss", "metric"):
                        history.setdefault(k, []).append(v)
                self.train()

                metric = val_info["metric"]
                if metric > best_metric:
                    best_metric = metric
                    epochs_no_improve = 0
                    best_state = copy.deepcopy(self.state_dict())
                else:
                    epochs_no_improve += 1

                self._log_epoch_with_val(epoch, epochs, epoch_train_loss, val_info, verbose_level, pbar)

                if patience is not None and epochs_no_improve >= patience:
                    if verbose_level >= 2:
                        self._log(
                            "early stopping at epoch %s (patience=%s)",
                            epoch + 1,
                            patience,
                            verbose=verbose_level,
                        )
                    break
            else:
                self._log_epoch_no_val(epoch, epochs, epoch_train_loss, verbose_level, pbar)

        if pbar is not None:
            pbar.close()

        if restore_best and best_state is not None:
            self.load_state_dict(best_state)

        history["stopped_epoch"] = stopped_epoch

        return history


__all__: list[str] = ["BaseTSK"]
