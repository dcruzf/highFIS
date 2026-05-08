"""Defuzzification strategies for normalized rule firing strengths.

This module defines defuzzifier classes used by TSK models to convert raw
antecedent firing strengths into normalized rule weights for consequent
aggregation. Each defuzzifier accepts an ``(N, R)`` tensor and returns a
normalized tensor of the same shape.

Built-in strategies:
    - ``SoftmaxLogDefuzzifier`` — ``softmax(log(w))``, numerically stable
      equivalent of ``w / sum(w)`` (default for HTSK and DG variants).
    - ``SumBasedDefuzzifier`` — classic ``w / sum(w)`` normalization
      (used by TSK, AYATSK, DombiTSK, AdaTSK).
    - ``LogSumDefuzzifier`` — temperature-scaled ``softmax(log(w) / T)``
      (used by LogTSK).
    - ``InvLogDefuzzifier`` — inverse-log normalization for log-domain
      firing strengths.

Notes:
    - A custom defuzzifier may be supplied via the ``defuzzifier`` constructor
      parameter of any model.
    - Custom defuzzifiers must accept a 2-D firing-strength tensor and return
      a normalized tensor of the same shape.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SoftmaxLogDefuzzifier(nn.Module):
    """Normalize firing strengths via ``softmax(log(w))``.

    Equivalent to ``w / sum(w)`` but numerically more stable in high
    dimensions thanks to the internal max-subtraction trick of
    :func:`torch.softmax`.
    """

    def __init__(self, eps: float | None = None) -> None:
        """Initialize with optional numeric stability floor.

        Args:
            eps: Small positive constant used to clamp weights before
                taking the logarithm.  ``None`` infers it from
                :func:`torch.finfo` for the input dtype at call time.
        """
        super().__init__()
        self.eps = eps

    def forward(self, w: Tensor) -> Tensor:
        """Normalize firing strengths via softmax(log(w)).

        Args:
            w: Firing-strength tensor of shape ``(N, R)`` with positive
                values.

        Returns:
            Normalized weight tensor of shape ``(N, R)`` summing to 1
            along the rule dimension.

        Raises:
            ValueError: If *w* does not have exactly 2 dimensions.
        """
        if w.ndim != 2:
            raise ValueError(f"expected w with 2 dims, got shape {tuple(w.shape)}")
        eps = self.eps if self.eps is not None else torch.finfo(w.dtype).eps
        log_w = w.clamp(min=eps).log()
        return torch.softmax(log_w, dim=1)


class SumBasedDefuzzifier(nn.Module):
    """Classic ``w / sum(w)`` normalization."""

    def __init__(self, eps: float | None = None) -> None:
        """Initialize with optional numeric stability floor.

        Args:
            eps: Small positive constant used to clamp weights before
                division.  ``None`` infers it from :func:`torch.finfo`
                for the input dtype at call time.
        """
        super().__init__()
        self.eps = eps

    def forward(self, w: Tensor) -> Tensor:
        """Normalize firing strengths via sum-based division.

        Args:
            w: Firing-strength tensor of shape ``(N, R)`` with positive
                values.

        Returns:
            Normalized weight tensor of shape ``(N, R)`` summing to 1
            along the rule dimension.

        Raises:
            ValueError: If *w* does not have exactly 2 dimensions.
        """
        if w.ndim != 2:
            raise ValueError(f"expected w with 2 dims, got shape {tuple(w.shape)}")
        eps = self.eps if self.eps is not None else torch.finfo(w.dtype).eps
        w = w.clamp(min=eps)
        return w / w.sum(dim=1, keepdim=True)


class LogSumDefuzzifier(nn.Module):
    """Normalize in log-space: ``softmax(log(w) / temperature)``.

    The *temperature* parameter controls the sharpness of the distribution.
    ``temperature=1`` recovers :class:`SoftmaxLogDefuzzifier`.
    """

    def __init__(self, temperature: float = 1.0, eps: float | None = None) -> None:
        """Initialize with temperature and optional numeric stability floor.

        Args:
            temperature: Positive scaling factor ``T > 0`` applied to the
                log-space weights before softmax.  ``1.0`` recovers
                :class:`SoftmaxLogDefuzzifier` behaviour; lower values
                sharpen the distribution, higher values flatten it.
            eps: Small positive constant used to clamp weights before
                taking the logarithm.  ``None`` infers it from
                :func:`torch.finfo` for the input dtype at call time.

        Raises:
            ValueError: If *temperature* is not positive.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.eps = eps

    def forward(self, w: Tensor) -> Tensor:
        """Normalize firing strengths via temperature-scaled softmax(log(w)).

        Args:
            w: Firing-strength tensor of shape ``(N, R)`` with positive
                values.

        Returns:
            Normalized weight tensor of shape ``(N, R)`` summing to 1
            along the rule dimension.

        Raises:
            ValueError: If *w* does not have exactly 2 dimensions.
        """
        if w.ndim != 2:
            raise ValueError(f"expected w with 2 dims, got shape {tuple(w.shape)}")
        eps = self.eps if self.eps is not None else torch.finfo(w.dtype).eps
        log_w = w.clamp(min=eps).log() / self.temperature
        return torch.softmax(log_w, dim=1)


class InvLogDefuzzifier(nn.Module):
    r"""Scale-invariant inverse-log defuzzifier for LogTSK.

    This defuzzifier normalizes log-domain firing strengths using an
    inverse-log transformation and L1 normalization.
    """

    def __init__(self, eps: float | None = None) -> None:
        """Initialize with numeric stability *eps*.

        Args:
            eps: Small constant used to clamp ``w`` away from zero and
                ``log(w)`` away from zero before inversion.  Defaults to
                :func:`torch.finfo` machine epsilon for the input dtype.
        """
        super().__init__()
        self.eps = eps

    def forward(self, w: Tensor) -> Tensor:
        r"""Compute L1-normalised inverse-log weights.

        Args:
            w: Tensor of shape ``(N, R)`` with positive firing strengths
                (product t-norm outputs), values in ``(0, 1]``.

        Returns:
            Tensor of shape ``(N, R)`` with normalized weights summing to 1.

        Raises:
            ValueError: If *w* does not have exactly 2 dimensions.
        """
        if w.ndim != 2:
            raise ValueError(f"expected w with 2 dims, got shape {tuple(w.shape)}")
        eps = self.eps if self.eps is not None else torch.finfo(w.dtype).eps
        # log_w ≤ 0; clamp away from 0 on both ends to avoid division by zero
        log_w = w.clamp(min=eps).log().clamp(max=-eps)  # strictly negative
        inv_log = -1.0 / log_w  # positive: 1/|Z_r|
        return inv_log / inv_log.sum(dim=1, keepdim=True)


__all__: list[str] = [
    "InvLogDefuzzifier",
    "LogSumDefuzzifier",
    "SoftmaxLogDefuzzifier",
    "SumBasedDefuzzifier",
]
