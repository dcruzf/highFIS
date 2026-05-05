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

    Note:
        This is a temperature-scaled softmax in log-space.  For the
        scale-invariant LogTSK defuzzifier from Du et al. (2020) /
        Cui et al. (2021), use :class:`InvLogDefuzzifier` instead.
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


class InvLogDefuzzifier(nn.Module):
    r"""Scale-invariant inverse-log defuzzifier for LogTSK (Du et al. 2020).

    Implements the L1-normalised inverse-log firing-strength formula:

    .. math::
        \bar{f}_r = \frac{1/|Z_r|}{\sum_{i=1}^{R} 1/|Z_i|}

    where :math:`Z_r = \log f_r = \sum_{d=1}^{D} \log \mu_{r,d} \leq 0` is
    the log-domain firing strength.  Because the weights depend on :math:`Z_r`
    only through its magnitude (not its scale), the output is **immune to
    softmax saturation** as the input dimension :math:`D` grows.

    References:
        Du, X. & Zeng, X.-J. (2020). "Fuzzy Rule-Based Classification System."
        Cui, Y., Wu, D. & Xu, Y. (2021). "Optimize TSK Fuzzy Systems for
        Regression Problems: Mini-Batch Gradient Descent with Regularization,
        DropRule, and AdaBound (MBGD-RDA)." *IEEE Trans. Fuzzy Syst.*
        29(5):1003-1015. §III-A.
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
