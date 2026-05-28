"""Single-phase gradient-based trainer (Adam)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor

from ..base import BaseTSK
from ._base import BaseTrainer


class GradientTrainer(BaseTrainer):
    """Single-phase mini-batch gradient descent trainer.

    Wraps :meth:`~highfis.base.BaseTSK.fit` with an explicit parameter set.
    This is the default trainer used by all standard highFIS estimators.

    Example::

        from highfis.optim import GradientTrainer

        trainer = GradientTrainer(epochs=200, learning_rate=1e-3)
        history = trainer.fit(model, x_train, y_train)
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
            verbose: Verbosity level passed to :meth:`~highfis.base.BaseTSK.fit`.
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
    ) -> dict[str, Any]:
        """Train *model* for :attr:`epochs` epochs and return the history dict."""
        return model.fit(
            x,
            y,
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            criterion=self.loss,
            batch_size=self.batch_size,
            shuffle=bool(self.shuffle),
            ur_weight=float(self.ur_weight),
            ur_target=self.ur_target,
            verbose=self.verbose,
            x_val=x_val,
            y_val=y_val,
            patience=self.patience,
            restore_best=bool(self.restore_best),
            weight_decay=float(self.weight_decay),
        )
