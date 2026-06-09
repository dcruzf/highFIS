"""Single-phase gradient-based trainer (Adam)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor
from tqdm.auto import trange

from ..base import BaseTSK
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

        has_val = model._validate_fit_inputs(x, y, x_val, y_val, self.ur_weight, self.ur_target)
        train_criterion = self.loss or model._default_criterion()
        train_optimizer = model._build_optimizer(optimizer, self.learning_rate, self.weight_decay)

        # Resolve/normalize metrics
        task = model._get_task()
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
            epoch_train_loss, epoch_ur_loss = model._run_minibatch_epoch(
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
                train_preds = model._predict_numpy(x)
                train_targets = y.cpu().numpy()
                train_metrics = compute_metrics(cast(Any, task), train_targets, train_preds, metrics=metrics_list)
                for m in metrics_list:
                    history.setdefault(f"train_{m}", []).append(train_metrics[m])

            if has_val and x_val is not None and y_val is not None:
                model.eval()
                val_info = model._evaluate_validation(train_criterion, x_val, y_val)
                history["val"].append(val_info["val_loss"])

                # Evaluate metrics on val set
                if metrics_list:
                    val_preds = model._predict_numpy(x_val)
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

                model._log_epoch_with_val(epoch, self.epochs, epoch_train_loss, val_info, verbose_level, pbar)

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
                model._log_epoch_no_val(epoch, self.epochs, epoch_train_loss, verbose_level, pbar)

        if pbar is not None:
            pbar.close()

        if self.restore_best and best_state is not None:
            model.load_state_dict(best_state)

        history["stopped_epoch"] = stopped_epoch

        return history
