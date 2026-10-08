"""Loss functions used by the families whose source articles define their own.

The families trained by plain gradient descent with a fixed learning rate depend on the
scale of the loss, because it sets the size of every step. Their articles sum the squared
error over the outputs, where ``torch.nn.MSELoss`` averages over them:

- ``SumSquaredErrorLoss`` — summed over the outputs, averaged over the samples
  (FSRE-ADATSK).
- ``HalfSumSquaredErrorLoss`` — the same, halved (DG-TSK).
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


class SumSquaredErrorLoss(nn.MSELoss):
    r"""Squared error summed over the outputs and averaged over the samples.

    $$E = \frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} (y_{n,c} - z_{n,c})^2$$

    This is the error function behind the gradients of the FSRE-ADATSK article (after its
    Eq. (8)). It is ``C`` times ``torch.nn.MSELoss`` for ``C`` outputs, so with three
    classes its gradient is three times larger. For a single output the two coincide.

    It subclasses ``MSELoss`` so that the trainers keep treating it as a squared-error
    loss, which for classifiers means comparing the outputs with one-hot targets.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Return ``sum((input - target) ** 2) / N`` for a batch of ``N`` samples."""
        return torch.sum((input - target) ** 2) / input.shape[0]


__all__: list[str] = ["HalfSumSquaredErrorLoss", "SumSquaredErrorLoss"]
