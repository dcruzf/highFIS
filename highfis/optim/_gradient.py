"""Single-phase gradient-based trainer (Adam)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import trange

from ..base import BaseTSK, _iter_minibatch_indices, _uniform_regularization_loss
from ._base import BaseTrainer


class GradientTrainer(BaseTrainer):
    """Single-phase mini-batch gradient descent trainer.

    This is the default trainer used by all standard highFIS estimators.

    Example::
        ```python
        from highfis.optim import GradientTrainer

        trainer = GradientTrainer(epochs=200, learning_rate=1e-3)
        history = trainer.fit(model, x_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        batch_size: int | None = 512,
        shuffle: bool = True,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        verbose: bool | int = False,
        loss: Callable[..., Any] | None = None,
    ) -> None:
        """Initialise a gradient trainer.

        Args:
            epochs: Maximum number of full passes over the training data.
            learning_rate: Initial learning rate for the Adam optimiser.
            batch_size: Mini-batch size. ``None`` uses the full dataset.
            shuffle: Reshuffle samples before each epoch.
            patience: Early-stopping patience.  ``None`` disables early
                stopping.
            restore_best: Restore the best validation weights after training.
            weight_decay: L2 weight-decay for consequent parameters.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target firing-level.
            verbose: Verbosity level.
            loss: Custom loss function ``f(output, target) -> scalar``.
                ``None`` uses the model's built-in criterion.
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.patience = patience
        self.restore_best = restore_best
        self.weight_decay = weight_decay
        self.ur_weight = ur_weight
        self.ur_target = ur_target
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
        metrics: list[str] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """Train *model* for :attr:`epochs` epochs and return the history dict."""
        from ..metrics import compute_metrics

        has_val = self._validate_fit_inputs(model, x, y, x_val, y_val, self.ur_weight, self.ur_target)
        train_criterion = self.loss or model.default_criterion()
        train_optimizer = self._build_optimizer(model, optimizer, self.learning_rate, self.weight_decay)

        # Resolve/normalize metrics
        task = getattr(model, "task_type", "regression")
        metrics_list: list[str] = []
        if metrics is not None:
            metrics_list = [metrics] if isinstance(metrics, str) else list(metrics)

        if metrics_list:
            primary_metric = metrics_list[0]
            _MAXIMIZE_METRICS = {
                "accuracy",
                "balanced_accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_micro",
                "recall_micro",
                "f1_micro",
                "r2",
                "explained_variance",
                "pearson",
            }
            maximize = primary_metric in _MAXIMIZE_METRICS

        history: dict[str, Any] = {"train": [], "ur": []}
        if has_val:
            history["val"] = []
        best_metric = float("-inf")
        epochs_no_improve = 0
        best_state: dict[str, Any] | None = None
        verbose_level = model._resolve_verbose(self.verbose)

        model.train()
        pbar = None
        if verbose_level == 1:
            pbar = trange(self.epochs, desc="Training", leave=False)
            epoch_iterator = pbar
        else:
            epoch_iterator = range(self.epochs)

        stopped_epoch = 0
        for epoch in epoch_iterator:
            stopped_epoch = epoch + 1
            epoch_train_loss, epoch_ur_loss = self._run_minibatch_epoch(
                model,
                x,
                y,
                train_criterion,
                train_optimizer,
                self.batch_size,
                self.shuffle,
                self.ur_weight,
                self.ur_target,
            )
            history["train"].append(epoch_train_loss)
            history["ur"].append(epoch_ur_loss)

            # Evaluate metrics on train set
            if metrics_list:
                train_preds = self._predict_numpy(model, x)
                train_targets = y.cpu().numpy()
                train_metrics = compute_metrics(cast(Any, task), train_targets, train_preds, metrics=metrics_list)
                for m in metrics_list:
                    history.setdefault(f"train_{m}", []).append(train_metrics[m])

            if has_val and x_val is not None and y_val is not None:
                model.eval()
                val_info = self._evaluate_validation(model, train_criterion, x_val, y_val)
                history["val"].append(val_info["val_loss"])

                # Evaluate metrics on val set
                if metrics_list:
                    val_preds = self._predict_numpy(model, x_val)
                    val_targets = y_val.cpu().numpy()
                    val_metrics = compute_metrics(cast(Any, task), val_targets, val_preds, metrics=metrics_list)
                    for m in metrics_list:
                        history.setdefault(f"val_{m}", []).append(val_metrics[m])
                        val_info[f"val_{m}"] = val_metrics[m]

                    # Update the metric used for early stopping
                    metric_val = val_metrics[primary_metric]
                    metric = metric_val if maximize else -metric_val
                    val_info["metric"] = metric
                else:
                    metric = val_info["metric"]

                for k, v in val_info.items():
                    if k not in ("val_loss", "metric") and k not in [f"val_{m}" for m in metrics_list]:
                        history.setdefault(k, []).append(v)
                model.train()

                if metric > best_metric:
                    best_metric = metric
                    epochs_no_improve = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve += 1

                self._log_epoch_with_val(model, epoch, self.epochs, epoch_train_loss, val_info, verbose_level, pbar)

                if self.patience is not None and epochs_no_improve >= self.patience:
                    if verbose_level >= 2:
                        model._log(
                            "early stopping at epoch %s (patience=%s)",
                            epoch + 1,
                            self.patience,
                            verbose=verbose_level,
                        )
                    break
            else:
                self._log_epoch_no_val(model, epoch, self.epochs, epoch_train_loss, verbose_level, pbar)

        if pbar is not None:
            pbar.close()

        if self.restore_best and best_state is not None:
            model.load_state_dict(best_state)

        history["stopped_epoch"] = stopped_epoch

        return history

    def _validate_fit_inputs(
        self,
        model: BaseTSK,
        x: Tensor,
        y: Tensor,
        x_val: Tensor | None,
        y_val: Tensor | None,
        ur_weight: float,
        ur_target: float | None,
    ) -> bool:
        """Validate all inputs to :meth:`fit` and return ``has_val``."""
        if x.ndim != 2 or x.shape[1] != model.n_inputs:
            raise ValueError(f"expected x shape (batch, {model.n_inputs}), got {tuple(x.shape)}")
        if y.ndim != 1:
            raise ValueError("expected y shape (batch,)")
        if ur_weight < 0.0:
            raise ValueError("ur_weight must be >= 0")
        if ur_target is not None and not (0.0 < ur_target <= 1.0):
            raise ValueError("ur_target must be in (0, 1] when provided")

        has_val = x_val is not None and y_val is not None
        if has_val and x_val is not None and y_val is not None:
            if x_val.ndim != 2 or x_val.shape[1] != model.n_inputs:
                raise ValueError(f"expected x_val shape (batch, {model.n_inputs}), got {tuple(x_val.shape)}")
            if y_val.ndim != 1:
                raise ValueError("expected y_val shape (batch,)")
        return has_val

    def _build_optimizer(
        self,
        model: BaseTSK,
        optimizer: torch.optim.Optimizer | None,
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        if optimizer is not None:
            return optimizer
        opt_class, param_groups = model._get_optimizer_config(learning_rate, weight_decay)
        return cast(Any, opt_class)(param_groups, lr=learning_rate)

    def _compute_loss(
        self,
        model: BaseTSK,
        criterion: Callable[[Tensor, Tensor], Tensor],
        output: Tensor,
        target: Tensor,
    ) -> Tensor:
        task = getattr(model, "task_type", "regression")
        if task == "classification":
            if isinstance(criterion, nn.MSELoss):
                one_hot = torch.zeros_like(output)
                one_hot.scatter_(1, target.unsqueeze(1), 1.0)
                return criterion(output, one_hot)
            return criterion(output, target)
        else:
            return criterion(output.squeeze(1), target)

    def _predict_numpy(self, model: BaseTSK, x: Tensor) -> np.ndarray:
        """Helper to get numpy predictions of the model on *x* using minibatch iteration."""
        task = getattr(model, "task_type", "regression")
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                outputs = []
                n_samples = x.shape[0]
                batch_size = 1024
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    out_b = model(x[start:end])
                    outputs.append(out_b)
                output = torch.cat(outputs, dim=0)
                if task == "classification":
                    return output.argmax(dim=1).cpu().numpy()
                else:
                    if output.ndim > 1 and output.shape[1] == 1:
                        output = output.squeeze(1)
                    return output.cpu().numpy()
        finally:
            model.train(was_training)

    def _run_minibatch_epoch(
        self,
        model: BaseTSK,
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
        for batch_idx in _iter_minibatch_indices(x.shape[0], batch_size=batch_size, shuffle=shuffle, device=x.device):
            x_b = x.index_select(0, batch_idx)
            y_b = y.index_select(0, batch_idx)

            optimizer.zero_grad(set_to_none=True)
            output, norm_w = model._forward_train(x_b)
            main_loss = self._compute_loss(model, criterion, output, y_b)

            ur_loss = _uniform_regularization_loss(norm_w, target=ur_target)
            total_loss = main_loss + float(ur_weight) * ur_loss
            total_loss.backward()
            optimizer.step()

            batch_losses.append(float(total_loss.detach().item()))
            batch_ur_losses.append(float(ur_loss.detach().item()))

        train_loss = float(sum(batch_losses) / max(len(batch_losses), 1))
        ur_loss_avg = float(sum(batch_ur_losses) / max(len(batch_ur_losses), 1))
        return train_loss, ur_loss_avg

    def _evaluate_validation(
        self,
        model: BaseTSK,
        criterion: Callable[[Tensor, Tensor], Tensor],
        x_val: Tensor,
        y_val: Tensor,
    ) -> dict[str, float]:
        """Evaluate on validation set.  Return dict with at least ``'metric'``."""
        task = getattr(model, "task_type", "regression")
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                outputs = []
                n_samples = x_val.shape[0]
                batch_size = 1024
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    out_b = model(x_val[start:end])
                    outputs.append(out_b)
                output = torch.cat(outputs, dim=0)
                val_loss = float(self._compute_loss(model, criterion, output, y_val).item())
                if task == "classification":
                    val_acc = float((output.argmax(dim=1) == y_val).float().mean().item())
                    return {"val_loss": val_loss, "val_acc": val_acc, "metric": val_acc}
                else:
                    return {"val_loss": val_loss, "metric": -val_loss}
        finally:
            model.train(was_training)

    def _log_epoch_with_val(
        self,
        model: BaseTSK,
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
            model._log(" ".join(log_parts), verbose=verbose_level)

    def _log_epoch_no_val(
        self,
        model: BaseTSK,
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
            model._log(
                "epoch=%s/%s loss=%.6f",
                epoch + 1,
                epochs,
                train_loss,
                verbose=verbose_level,
            )
