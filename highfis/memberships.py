from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _inv_softplus(value: float, eps: float | None = None) -> float:
    """Map a positive value to the unconstrained space used by softplus."""
    eps = torch.finfo(torch.get_default_dtype()).eps if eps is None else eps
    v = max(value - eps, eps)
    return math.log(math.expm1(v))


class MembershipFunction(nn.Module):
    """Base class for differentiable membership functions in PyTorch."""

    def __init__(self, eps: float | None = None) -> None:
        """Initialize base membership function with numeric stability epsilon."""
        super().__init__()
        self.eps = torch.finfo(torch.get_default_dtype()).eps if eps is None else float(eps)

    def _as_tensor(self, x: Tensor | float) -> Tensor:
        if isinstance(x, Tensor):
            return x
        return torch.as_tensor(x, dtype=torch.get_default_dtype())


class GaussianMF(MembershipFunction):
    """Gaussian membership: exp(-((x-c)^2)/(2*sigma^2))."""

    def __init__(self, mean: float = 0.0, sigma: float = 1.0, eps: float | None = None) -> None:
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


class CompositeGaussianMF(MembershipFunction):
    """Composite Gaussian membership with a nonzero lower bound."""

    def __init__(self, mean: float = 0.0, sigma: float = 1.0, eps: float | None = None) -> None:
        """Initialize composite Gaussian membership parameters."""
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
        """Compute Composite Gaussian membership values for input tensor."""
        x = self._as_tensor(x)
        z = (x - self.mean) / self.sigma
        return self.eps + (1.0 - self.eps) * torch.exp(-0.5 * z.square())


class TriangularMF(MembershipFunction):
    """Triangular membership function defined by left, center, right."""

    def __init__(self, left: float = -1.0, center: float = 0.0, right: float = 1.0, eps: float | None = None) -> None:
        """Initialize triangular MF with *left*, *center*, *right* vertices."""
        super().__init__(eps=eps)
        if not (left <= center <= right):
            raise ValueError("must satisfy left <= center <= right")
        if left == right:
            raise ValueError("left and right must differ")
        self.left = nn.Parameter(torch.tensor(float(left)))
        self.center = nn.Parameter(torch.tensor(float(center)))
        self.right = nn.Parameter(torch.tensor(float(right)))

    def forward(self, x: Tensor) -> Tensor:
        """Compute triangular membership values for input tensor."""
        x = self._as_tensor(x)
        left_slope = (x - self.left) / (self.center - self.left + self.eps)
        right_slope = (self.right - x) / (self.right - self.center + self.eps)
        return torch.clamp(torch.minimum(left_slope, right_slope), min=0.0, max=1.0)


class TrapezoidalMF(MembershipFunction):
    """Trapezoidal membership function defined by a, b, c, d."""

    def __init__(
        self, a: float = -2.0, b: float = -1.0, c: float = 1.0, d: float = 2.0, eps: float | None = None
    ) -> None:
        """Initialize trapezoidal MF with vertices *a*, *b*, *c*, *d*."""
        super().__init__(eps=eps)
        if not (a <= b <= c <= d):
            raise ValueError("must satisfy a <= b <= c <= d")
        if a == d:
            raise ValueError("a and d must differ")
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.b = nn.Parameter(torch.tensor(float(b)))
        self.c = nn.Parameter(torch.tensor(float(c)))
        self.d = nn.Parameter(torch.tensor(float(d)))

    def forward(self, x: Tensor) -> Tensor:
        """Compute trapezoidal membership values for input tensor."""
        x = self._as_tensor(x)
        left = (x - self.a) / (self.b - self.a + self.eps)
        right = (self.d - x) / (self.d - self.c + self.eps)
        return torch.clamp(torch.minimum(left, right), min=0.0, max=1.0)


class BellMF(MembershipFunction):
    """Generalized bell membership: 1 / (1 + |((x-c)/a)|^(2b))."""

    def __init__(self, a: float = 1.0, b: float = 2.0, center: float = 0.0, eps: float | None = None) -> None:
        """Initialize bell MF with width *a*, slope *b*, and *center*."""
        super().__init__(eps=eps)
        if a <= 0:
            raise ValueError("a must be positive")
        if b <= 0:
            raise ValueError("b must be positive")
        self.raw_a = nn.Parameter(torch.tensor(_inv_softplus(float(a), eps)))
        self.raw_b = nn.Parameter(torch.tensor(_inv_softplus(float(b), eps)))
        self.center = nn.Parameter(torch.tensor(float(center)))

    @property
    def a(self) -> Tensor:
        """Return positive width via softplus reparameterization."""
        return F.softplus(self.raw_a) + self.eps

    @property
    def b(self) -> Tensor:
        """Return positive slope via softplus reparameterization."""
        return F.softplus(self.raw_b) + self.eps

    def forward(self, x: Tensor) -> Tensor:
        """Compute bell membership values for input tensor."""
        x = self._as_tensor(x)
        return 1.0 / (1.0 + torch.abs((x - self.center) / self.a).pow(2.0 * self.b))


class SigmoidalMF(MembershipFunction):
    """Sigmoidal membership: 1 / (1 + exp(-a*(x-c)))."""

    def __init__(self, a: float = 1.0, center: float = 0.0, eps: float | None = None) -> None:
        """Initialize sigmoidal MF with slope *a* and *center*."""
        super().__init__(eps=eps)
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.center = nn.Parameter(torch.tensor(float(center)))

    def forward(self, x: Tensor) -> Tensor:
        """Compute sigmoidal membership values for input tensor."""
        x = self._as_tensor(x)
        return torch.sigmoid(self.a * (x - self.center))


__all__: list[str] = [
    "BellMF",
    "CompositeGaussianMF",
    "GaussianMF",
    "MembershipFunction",
    "SigmoidalMF",
    "TrapezoidalMF",
    "TriangularMF",
]
