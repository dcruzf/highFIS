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
            model: The BaseTSK model to train.
            x: Training input tensor of shape ``(n_samples, n_features)``.
            y: Training target tensor.
            x_val: Optional validation input tensor.
            y_val: Optional validation target tensor.

        Returns:
            A dictionary with training history.  Keys depend on the concrete
            implementation. GradientTrainer returns the history
            dict from BaseTSK.fit(). DGTrainer
            returns a dict with keys ``"dg"``, ``"threshold"``, and
            ``"finetune"``.
        """
