"""Defuzzification strategies for normalized rule firing strengths."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SoftmaxLogDefuzzifier(nn.Module):
    """Normalize firing strengths via ``softmax(log(w))``.

    Mathematically equivalent to ``w / sum(w)`` but numerically more stable
    in high dimensions thanks to the internal max-subtraction trick of
    :func:`torch.softmax`.
    """

    def __init__(self, eps: float | None = None) -> None:
        """Initialize with numeric stability *eps*."""
        super().__init__()
        self.eps = eps

    def forward(self, w: Tensor) -> Tensor:
        """Normalize firing strengths via softmax(log(w))."""
        if w.ndim != 2:
            raise ValueError(f"expected w with 2 dims, got shape {tuple(w.shape)}")
        eps = self.eps if self.eps is not None else torch.finfo(w.dtype).eps
        log_w = w.clamp(min=eps).log()
        return torch.softmax(log_w, dim=1)


class SumBasedDefuzzifier(nn.Module):
    """Classic ``w / sum(w)`` normalization."""

    def __init__(self, eps: float | None = None) -> None:
        """Initialize with numeric stability *eps*."""
        super().__init__()
        self.eps = eps

    def forward(self, w: Tensor) -> Tensor:
        """Normalize firing strengths via sum-based division."""
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
        """Initialize with *temperature* and numeric stability *eps*."""
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.eps = eps

    def forward(self, w: Tensor) -> Tensor:
        """Normalize firing strengths via temperature-scaled softmax(log(w))."""
        if w.ndim != 2:
            raise ValueError(f"expected w with 2 dims, got shape {tuple(w.shape)}")
        eps = self.eps if self.eps is not None else torch.finfo(w.dtype).eps
        log_w = w.clamp(min=eps).log() / self.temperature
        return torch.softmax(log_w, dim=1)


__all__: list[str] = ["LogSumDefuzzifier", "SoftmaxLogDefuzzifier", "SumBasedDefuzzifier"]
