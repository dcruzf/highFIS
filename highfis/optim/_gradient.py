"""Single-phase gradient-based trainer (Adam)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

from ..metrics import compute_metrics
from ..models._base import BaseTSK, set_training_flag
from ._base import BaseTrainer
from ._utils import (
    _get_optimizer_config,
    _log,
    _resolve_verbose,
    _uniform_regularization_loss,
)

_MAXIMIZE_METRICS: frozenset[str] = frozenset(
    {
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
)


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
        eval_metrics_every: int = 1,
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
                - ``False`` / ``0``: silent.
                - ``True``  / ``1``: progress bar (tqdm).
                - ``2``: log ~10 points during training.
                - ``3``: log every epoch.
            loss: Custom loss function ``f(output, target) -> scalar``.
                ``None`` uses the model's built-in criterion.
            eval_metrics_every: Evaluate the *training* metrics every ``n`` epochs;
                ``0`` skips them entirely. Each evaluation is a full extra forward pass
                over the training set, and the resulting ``history["train_<metric>"]``
                entries are diagnostic only -- they never feed early stopping, which
                consults validation metrics alone. Validation metrics are unaffected and
                are always evaluated every epoch.
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
        self.eval_metrics_every = eval_metrics_every

    def _should_eval_train_metrics(self, epoch: int) -> bool:
        """Whether the training-metric pass runs on this (0-based) epoch."""
        every = int(self.eval_metrics_every)
        return every > 0 and (epoch + 1) % every == 0

    def _init_history(self, has_val: bool, metrics_list: list[str]) -> dict[str, Any]:
        """Initialize history dictionary with a fixed schema."""
        history: dict[str, Any] = {
            "config": {
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "ur_weight": self.ur_weight,
                "ur_target": self.ur_target,
            },
            "stopped_epoch": 0,
            "train": [],
            "ur": [],
            "train_loss": [],
            "train_ur_loss": [],
            "train_total_loss": [],
            "val_loss": [] if has_val else None,
            "lr": [],
        }
        if has_val:
            history["val"] = []
        for m in metrics_list:
            if int(self.eval_metrics_every) > 0:
                history[f"train_{m}"] = []
            if has_val:
                history[f"val_{m}"] = []
        return history

    def _evaluate_epoch_metrics(
        self,
        model: BaseTSK,
        x: Tensor,
        y: Tensor,
        metrics_list: list[str],
        history: dict[str, Any],
        prefix: str,
        preds: Tensor | None = None,
    ) -> dict[str, float]:
        if preds is None:
            preds = self._predict_tensor(model, x)
        m_vals = compute_metrics(cast(Any, model.task_type), y, preds, metrics=metrics_list)
        for m in metrics_list:
            history.setdefault(f"{prefix}_{m}", []).append(m_vals[m])
        return m_vals

    def _handle_validation_epoch(
        self,
        model: BaseTSK,
        x_val: Tensor,
        y_val: Tensor,
        metrics_list: list[str],
        maximize: bool,
        train_criterion: Any,
        history: dict[str, Any],
        best_metric: float,
        epochs_no_improve: int,
        best_state: dict[str, Any] | None,
        verbose_level: int,
        epoch: int,
        epoch_train_loss: float,
        pbar: Any,
    ) -> tuple[float, int, dict[str, Any] | None, bool]:
        """Run validation evaluation, logging, and early-stopping logic."""
        set_training_flag(model, False)
        # One forward pass feeds both the loss and the metrics; they are evaluated under
        # the same weights and the same eval mode, so recomputing it would be pure waste.
        val_output = self._forward_batched(model, x_val)
        val_info = self._evaluate_validation(model, train_criterion, x_val, y_val, output=val_output)
        history["val"].append(val_info["val_loss"])
        history["val_loss"].append(val_info["val_loss"])

        if metrics_list:
            val_preds = self._preds_from_output(model, val_output)
            val_metrics = self._evaluate_epoch_metrics(
                model, x_val, y_val, metrics_list, history, "val", preds=val_preds
            )
            for m in metrics_list:
                val_info[f"val_{m}"] = val_metrics[m]

            primary_metric = metrics_list[0]
            metric_val = val_metrics[primary_metric]
            metric = metric_val if maximize else -metric_val
            val_info["metric"] = metric
        else:
            metric = val_info["metric"]

        # val_accuracy de classificação: salvar somente se "accuracy" não está nas métricas explícitas
        if "val_accuracy" in val_info and "accuracy" not in metrics_list:
            history.setdefault("val_accuracy", []).append(val_info["val_accuracy"])
            history.setdefault("val_acc", []).append(val_info["val_accuracy"])

        set_training_flag(model, True)

        should_stop = False
        if metric > best_metric:
            best_metric = metric
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        self._log_epoch_with_val(model, epoch, self.epochs, epoch_train_loss, val_info, verbose_level, pbar)

        if self.patience is not None and epochs_no_improve >= self.patience:
            if verbose_level >= 2:
                _log(
                    model.logger,
                    "early stopping at epoch %s (patience=%s)",
                    epoch + 1,
                    self.patience,
                    verbose=verbose_level,
                )
            should_stop = True

        return best_metric, epochs_no_improve, best_state, should_stop

    def _resolve_metrics(self, metrics: list[str] | str | None, task: str) -> tuple[list[str], bool]:
        """Resolve and normalize task metrics, returning metrics_list and maximize flag."""
        if metrics is None:
            metrics_list = ["accuracy"] if task == "classification" else ["mse"]
        else:
            metrics_list = [metrics] if isinstance(metrics, str) else list(metrics)

        maximize = False
        if metrics_list:
            primary_metric = metrics_list[0]
            maximize = primary_metric in _MAXIMIZE_METRICS
        return metrics_list, maximize

    def _build_train_loader(self, x: Tensor, y: Tensor) -> DataLoader:
        """Create DataLoader for training."""
        dataset = TensorDataset(x, y)
        actual_batch_size = len(dataset) if self.batch_size is None else self.batch_size
        return DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=self.shuffle,
        )

    def _get_epoch_iterator(self, verbose_level: int) -> tuple[Any, Any]:
        """Retrieve the loop iterator and optional progress bar."""
        pbar = None
        if verbose_level == 1:
            pbar = trange(self.epochs, desc="Training", leave=False)
            epoch_iterator = pbar
        else:
            epoch_iterator = range(self.epochs)
        return epoch_iterator, pbar

    def _finalize_training(self, model: BaseTSK, best_state: dict[str, Any] | None, pbar: Any) -> None:
        """Perform final training cleanup and best-state restoration."""
        if pbar is not None:
            pbar.close()

        if self.restore_best and best_state is not None:
            model.load_state_dict(best_state)

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
        scheduler: torch.optim.lr_scheduler.LRScheduler | Any = None,
    ) -> dict[str, Any]:
        """Train *model* for :attr:`epochs` epochs and return the history dict."""
        has_val = self._validate_fit_inputs(model, x, y, x_val, y_val, self.ur_weight, self.ur_target)
        train_criterion = self.loss or model.default_criterion()
        train_optimizer = self._build_optimizer(model, optimizer, self.learning_rate, self.weight_decay)

        train_loader = self._build_train_loader(x, y)
        metrics_list, maximize = self._resolve_metrics(metrics, model.task_type)

        history = self._init_history(has_val, metrics_list)
        best_metric = float("-inf")
        epochs_no_improve = 0
        best_state: dict[str, Any] | None = None
        verbose_level = _resolve_verbose(self.verbose)

        model.train()
        epoch_iterator, pbar = self._get_epoch_iterator(verbose_level)

        for epoch in epoch_iterator:
            history["stopped_epoch"] = epoch + 1
            epoch_main_loss, epoch_ur_loss, epoch_total_loss = self._run_minibatch_epoch(
                model,
                train_loader,
                train_criterion,
                train_optimizer,
                self.ur_weight,
                self.ur_target,
            )
            history["train"].append(epoch_total_loss)
            history["ur"].append(epoch_ur_loss)
            history["train_loss"].append(epoch_main_loss)
            history["train_ur_loss"].append(epoch_ur_loss)
            history["train_total_loss"].append(epoch_total_loss)

            # Evaluate metrics on train set
            if metrics_list and self._should_eval_train_metrics(epoch):
                self._evaluate_epoch_metrics(model, x, y, metrics_list, history, "train")

            if has_val and x_val is not None and y_val is not None:
                best_metric, epochs_no_improve, best_state, should_stop = self._handle_validation_epoch(
                    model,
                    x_val,
                    y_val,
                    metrics_list,
                    maximize,
                    train_criterion,
                    history,
                    best_metric,
                    epochs_no_improve,
                    best_state,
                    verbose_level,
                    epoch,
                    epoch_total_loss,
                    pbar,
                )
                if should_stop:
                    break
            else:
                self._log_epoch_no_val(model, epoch, self.epochs, epoch_total_loss, verbose_level, pbar)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    val_loss_val = history["val_loss"][-1] if (has_val and history["val_loss"]) else epoch_main_loss
                    scheduler.step(val_loss_val)
                else:
                    scheduler.step()

            current_lr = train_optimizer.param_groups[0]["lr"]
            history["lr"].append(current_lr)

        self._finalize_training(model, best_state, pbar)

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
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be > 0 when provided")

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
        opt_class, param_groups = _get_optimizer_config(model, learning_rate, weight_decay)
        return cast(Any, opt_class)(param_groups, lr=learning_rate)

    def _compute_loss(
        self,
        model: BaseTSK,
        criterion: Callable[[Tensor, Tensor], Tensor],
        output: Tensor,
        target: Tensor,
    ) -> Tensor:
        task = model.task_type
        if task == "classification":
            if isinstance(criterion, nn.MSELoss):
                one_hot = torch.zeros_like(output)
                one_hot.scatter_(1, target.unsqueeze(1), 1.0)
                return criterion(output, one_hot)
            return criterion(output, target)
        else:
            return criterion(output.squeeze(1), target)

    def _forward_batched(self, model: BaseTSK, x: Tensor) -> Tensor:
        """Execute forward pass in mini-batches without gradients. Restore training mode."""
        was_training = model.training
        set_training_flag(model, False)
        try:
            with torch.no_grad():
                outputs = []
                loader = DataLoader(TensorDataset(x), batch_size=1024, shuffle=False)
                for (x_b,) in loader:
                    outputs.append(model(x_b))
                return torch.cat(outputs, dim=0)
        finally:
            set_training_flag(model, was_training)

    def _preds_from_output(self, model: BaseTSK, output: Tensor) -> Tensor:
        """Reduce raw forward outputs to predictions for the model's task."""
        if model.task_type == "classification":
            return output.argmax(dim=1)
        return output.squeeze(1) if output.ndim > 1 and output.shape[1] == 1 else output

    def _predict_tensor(self, model: BaseTSK, x: Tensor) -> Tensor:
        """Helper to get PyTorch Tensor predictions of the model on *x* using DataLoader."""
        return self._preds_from_output(model, self._forward_batched(model, x))

    def _predict_numpy(self, model: BaseTSK, x: Tensor) -> np.ndarray:
        """Helper to get numpy predictions of the model on *x* using DataLoader."""
        return self._predict_tensor(model, x).cpu().numpy()

    def _run_minibatch_epoch(
        self,
        model: BaseTSK,
        loader: DataLoader,
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        ur_weight: float,
        ur_target: float | None,
    ) -> tuple[float, float, float]:
        """Run one full epoch of mini-batch gradient updates.

        Returns:
            ``(mean_main_loss, mean_ur_loss, mean_total_loss)`` averaged over all batches.
        """
        batch_main_losses: list[float] = []
        batch_ur_losses: list[float] = []
        batch_total_losses: list[float] = []
        for x_b, y_b in loader:
            optimizer.zero_grad(set_to_none=True)
            output, norm_w = model._forward_train(x_b)
            main_loss = self._compute_loss(model, criterion, output, y_b)

            ur_loss = _uniform_regularization_loss(norm_w, target=ur_target)
            total_loss = main_loss + float(ur_weight) * ur_loss
            total_loss.backward()
            optimizer.step()

            batch_main_losses.append(float(main_loss.detach().item()))
            batch_ur_losses.append(float(ur_loss.detach().item()))
            batch_total_losses.append(float(total_loss.detach().item()))

        def mean(lst: list[float]) -> float:
            return float(sum(lst) / max(len(lst), 1))

        return mean(batch_main_losses), mean(batch_ur_losses), mean(batch_total_losses)

    def _evaluate_validation(
        self,
        model: BaseTSK,
        criterion: Callable[[Tensor, Tensor], Tensor],
        x_val: Tensor,
        y_val: Tensor,
        output: Tensor | None = None,
    ) -> dict[str, float]:
        """Evaluate on validation set.  Return dict with at least ``'metric'``.

        *output* lets the caller supply a forward pass it has already run, so the
        per-epoch validation metrics do not repeat it.
        """
        if output is None:
            output = self._forward_batched(model, x_val)
        val_loss = float(self._compute_loss(model, criterion, output, y_val).item())
        if model.task_type == "classification":
            val_acc = float((output.argmax(dim=1) == y_val).float().mean().item())
            return {"val_loss": val_loss, "val_accuracy": val_acc, "metric": val_acc}
        return {"val_loss": val_loss, "metric": -val_loss}

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
            _log(model.logger, " ".join(log_parts), verbose=verbose_level)

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
            _log(
                model.logger,
                "epoch=%s/%s loss=%.6f",
                epoch + 1,
                epochs,
                train_loss,
                verbose=verbose_level,
            )
