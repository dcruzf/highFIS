from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _inv_softplus(value: float, eps: float = 1e-6) -> float:
    """Map a positive value to the unconstrained space used by softplus."""
    v = max(value - eps, eps)
    return math.log(math.expm1(v))


class MembershipFunction(nn.Module):
    """Base class for differentiable membership functions in PyTorch."""

    def __init__(self, eps: float = 1e-6) -> None:
        """Initialize base membership function with numeric stability epsilon."""
        super().__init__()
        self.eps = eps

    def _as_tensor(self, x: Tensor | float) -> Tensor:
        if isinstance(x, Tensor):
            return x
        return torch.as_tensor(x, dtype=torch.get_default_dtype())


class GaussianMF(MembershipFunction):
    """Gaussian membership: exp(-((x-c)^2)/(2*sigma^2))."""

    def __init__(self, mean: float = 0.0, sigma: float = 1.0, eps: float = 1e-6) -> None:
        """Initialize Gaussian membership parameters."""
        super().__init__(eps=eps)
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mean = nn.Parameter(torch.tensor(float(mean)))
        self.raw_sigma = nn.Parameter(torch.tensor(_inv_softplus(float(sigma), eps)))

    @property
    def sigma(self) -> Tensor:
        """Return positive sigma using softplus reparameterization."""
        return F.softplus(self.raw_sigma) + self.eps

    def forward(self, x: Tensor) -> Tensor:
        """Compute Gaussian membership values for input tensor."""
        x = self._as_tensor(x)
        z = (x - self.mean) / self.sigma
        return torch.exp(-0.5 * z.square())


__all__ = ["MembershipFunction", "GaussianMF"]
