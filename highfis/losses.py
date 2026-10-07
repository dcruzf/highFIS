"""Loss functions used by the families whose source articles define their own.

``HalfSumSquaredErrorLoss`` is the loss of the gated families: the squared error summed
over the outputs, averaged over the samples and halved. It differs from
``torch.nn.MSELoss`` by a constant factor, which changes the size of each step under plain
gradient descent with a fixed learning rate.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HalfSumSquaredErrorLoss(nn.MSELoss):
    r"""Squared error summed over the outputs, averaged over the samples and halved.

    $$E = \frac{1}{2N} \sum_{n=1}^{N} \sum_{c=1}^{C} (y_{n,c} - z_{n,c})^2$$

    This is the loss of the DG-TSK reference implementation and of the DG-ALETSK article
    (Eq. (28)). ``torch.nn.MSELoss`` also divides by the number of outputs ``C``, so the
    two differ by the factor ``C / 2``: with three classes its gradient is 1.5 times
    smaller.

    It subclasses ``MSELoss`` so that the trainers keep treating it as a squared-error
    loss, which for classifiers means comparing the outputs with one-hot targets.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Return ``sum((input - target) ** 2) / (2 * N)`` for a batch of ``N`` samples."""
        return torch.sum((input - target) ** 2) / (2.0 * input.shape[0])


__all__: list[str] = ["HalfSumSquaredErrorLoss"]
