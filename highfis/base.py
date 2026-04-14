"""Base TSK model that factors the common antecedent-defuzzification pipeline."""

from __future__ import annotations

import copy
import logging
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from .defuzzifiers import SoftmaxLogDefuzzifier
from .layers import MembershipLayer, RuleLayer
from .memberships import MembershipFunction
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
        t_norm: str = "gmean",
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize the TSK pipeline layers."""
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
            t_norm=t_norm if t_norm_fn is None else "prod",
            t_norm_fn=t_norm_fn,
        )
        self.n_rules = self.rule_layer.n_rules
        self.defuzzifier = defuzzifier or SoftmaxLogDefuzzifier()
        self.consequent_batch_norm = bool(consequent_batch_norm)
        self.consequent_bn = nn.BatchNorm1d(self.n_inputs) if self.consequent_batch_norm else None
        self.consequent_layer = self._build_consequent_layer()

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
        return cast(Tensor, self.defuzzifier(w))

    def _forward_train(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass returning ``(output, norm_w)`` to avoid double computation."""
        mu = self.membership_layer(x)
        w = self.rule_layer(mu)
        norm_w = self.defuzzifier(w)
        x_cons = self.consequent_bn(x) if self.consequent_bn is not None else x
        output = cast(Tensor, self.consequent_layer(x_cons, norm_w))
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
        verbose: bool = False,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
        patience: int = 20,
        weight_decay: float = 1e-8,
    ) -> dict[str, Any]:
        """Train the model with optional early stopping.

        When *x_val* and *y_val* are provided the model evaluates a
        task-specific metric (via :meth:`_evaluate_validation`) after every
        epoch and applies early stopping when the metric has not improved for
        *patience* consecutive epochs.
        """
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"expected x shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
        if y.ndim != 1:
            raise ValueError("expected y shape (batch,)")
        if ur_weight < 0.0:
            raise ValueError("ur_weight must be >= 0")
        if ur_target is not None and not (0.0 < ur_target <= 1.0):
            raise ValueError("ur_target must be in (0, 1] when provided")

        has_val = x_val is not None and y_val is not None
        if has_val:
            if x_val is None or y_val is None:  # pragma: no cover
                raise ValueError("x_val and y_val must both be provided")
            if x_val.ndim != 2 or x_val.shape[1] != self.n_inputs:
                raise ValueError(f"expected x_val shape (batch, {self.n_inputs}), got {tuple(x_val.shape)}")
            if y_val.ndim != 1:
                raise ValueError("expected y_val shape (batch,)")

        train_criterion = criterion or self._default_criterion()
        if optimizer is not None:
            train_optimizer = optimizer
        else:
            ante_params = list(self.membership_layer.parameters())
            cons_params = list(self.consequent_layer.parameters())
            if self.consequent_bn is not None:
                cons_params.extend(self.consequent_bn.parameters())
            train_optimizer = torch.optim.AdamW(
                [
                    {"params": ante_params, "weight_decay": 0.0},
                    {"params": cons_params, "weight_decay": weight_decay},
                ],
                lr=learning_rate,
            )

        history: dict[str, Any] = {"train": [], "ur": [], "val": []}
        best_metric = float("-inf")
        epochs_no_improve = 0
        best_state: dict[str, Any] | None = None

        self.train()
        for epoch in range(epochs):
            batch_losses: list[float] = []
            batch_ur_losses: list[float] = []
            for batch_idx in _iter_minibatch_indices(x.shape[0], batch_size=batch_size, shuffle=shuffle):
                x_b = x.index_select(0, batch_idx.to(device=x.device))
                y_b = y.index_select(0, batch_idx.to(device=y.device))

                train_optimizer.zero_grad(set_to_none=True)
                output, norm_w = self._forward_train(x_b)
                main_loss = self._compute_loss(train_criterion, output, y_b)

                ur_loss = _uniform_regularization_loss(norm_w, target=ur_target)
                loss = main_loss + (float(ur_weight) * ur_loss)
                loss.backward()
                train_optimizer.step()

                batch_losses.append(float(loss.detach().item()))
                batch_ur_losses.append(float(ur_loss.detach().item()))

            epoch_train_loss = float(sum(batch_losses) / max(len(batch_losses), 1))
            history["train"].append(epoch_train_loss)
            history["ur"].append(float(sum(batch_ur_losses) / max(len(batch_ur_losses), 1)))

            # --- validation & early stopping ---
            if has_val and x_val is not None and y_val is not None:
                self.eval()
                val_info = self._evaluate_validation(train_criterion, x_val, y_val)
                history["val"].append(val_info.get("val_loss", 0.0))
                # Store any extra keys (e.g. val_acc) in history
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

                if verbose and ((epoch + 1) % max(epochs // 10, 1) == 0 or epoch == 0):
                    log_parts = [f"epoch={epoch + 1}/{epochs}", f"train_loss={epoch_train_loss:.6f}"]
                    for k, v in val_info.items():
                        if k != "metric":
                            log_parts.append(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")
                    logger.info(" ".join(log_parts))

                if epochs_no_improve >= patience:
                    if verbose:
                        logger.info("early stopping at epoch %s (patience=%s)", epoch + 1, patience)
                    break
            else:
                if verbose and ((epoch + 1) % max(epochs // 10, 1) == 0 or epoch == 0):
                    logger.info("epoch=%s/%s loss=%.6f", epoch + 1, epochs, epoch_train_loss)

        if best_state is not None:
            self.load_state_dict(best_state)

        history["stopped_epoch"] = epoch + 1  # type: ignore[possibly-undefined]

        return history


__all__: list[str] = ["BaseTSK"]
