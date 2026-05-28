"""Abstract base class for highFIS trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from torch import Tensor

if TYPE_CHECKING:
    from ..base import BaseTSK


class BaseTrainer(ABC):
    """Abstract base class for training strategies.

    A trainer encapsulates the optimisation loop and is decoupled from the
    sklearn estimator.  Concrete subclasses implement the full training
    protocol — e.g., single-phase gradient descent, three-phase DG training,
    or hybrid LSE / gradient procedures.

    Usage::

        from highfis.optim import GradientTrainer

        trainer = GradientTrainer(epochs=100, learning_rate=1e-3)
        history = trainer.fit(model, x_train, y_train, x_val=x_val, y_val=y_val)
    """

    @abstractmethod
    def fit(
        self,
        model: BaseTSK,
        x: Tensor,
        y: Tensor,
        *,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
    ) -> dict[str, Any]:
        """Train *model* on *(x, y)* and return a history dictionary.

        Args:
            model: The :class:`~highfis.base.BaseTSK` model to train.
            x: Training input tensor of shape ``(n_samples, n_features)``.
            y: Training target tensor.
            x_val: Optional validation input tensor.
            y_val: Optional validation target tensor.

        Returns:
            A dictionary with training history.  Keys depend on the concrete
            implementation.  :class:`GradientTrainer` returns the history
            dict from :meth:`~highfis.base.BaseTSK.fit`.  :class:`DGTrainer`
            returns a dict with keys ``"dg"``, ``"threshold"``, and
            ``"finetune"``.
        """
