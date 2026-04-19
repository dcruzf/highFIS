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


def _smoothstep(t: Tensor) -> Tensor:
    """Cubic smoothstep S(t)=3t^2-2t^3 for t in [0, 1]."""
    return 3.0 * t.square() - 2.0 * t.pow(3)


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


class DiffSigmoidalMF(MembershipFunction):
    """Difference of two sigmoids: s1(x) - s2(x)."""

    def __init__(self, a1: float, center1: float, a2: float, center2: float, eps: float | None = None) -> None:
        """Initialize with slope and center parameters for each sigmoid."""
        super().__init__(eps=eps)
        self.a1 = nn.Parameter(torch.tensor(float(a1)))
        self.center1 = nn.Parameter(torch.tensor(float(center1)))
        self.a2 = nn.Parameter(torch.tensor(float(a2)))
        self.center2 = nn.Parameter(torch.tensor(float(center2)))

    def forward(self, x: Tensor) -> Tensor:
        """Compute s1(x) - s2(x) membership values for input tensor."""
        x = self._as_tensor(x)
        s1 = torch.sigmoid(self.a1 * (x - self.center1))
        s2 = torch.sigmoid(self.a2 * (x - self.center2))
        return s1 - s2


class ProdSigmoidalMF(MembershipFunction):
    """Product of two sigmoids: s1(x) * s2(x)."""

    def __init__(self, a1: float, center1: float, a2: float, center2: float, eps: float | None = None) -> None:
        """Initialize with slope and center parameters for each sigmoid."""
        super().__init__(eps=eps)
        self.a1 = nn.Parameter(torch.tensor(float(a1)))
        self.center1 = nn.Parameter(torch.tensor(float(center1)))
        self.a2 = nn.Parameter(torch.tensor(float(a2)))
        self.center2 = nn.Parameter(torch.tensor(float(center2)))

    def forward(self, x: Tensor) -> Tensor:
        """Compute s1(x) * s2(x) membership values for input tensor."""
        x = self._as_tensor(x)
        s1 = torch.sigmoid(self.a1 * (x - self.center1))
        s2 = torch.sigmoid(self.a2 * (x - self.center2))
        return s1 * s2


class SShapedMF(MembershipFunction):
    """Smooth S-shaped membership from 0 to 1 between a and b."""

    def __init__(self, a: float, b: float, eps: float | None = None) -> None:
        """Initialize with transition start (a) and end (b); requires a < b."""
        super().__init__(eps=eps)
        if not (a < b):
            raise ValueError("expected a < b")
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.raw_d = nn.Parameter(torch.tensor(_inv_softplus(float(b - a), eps)))

    @property
    def b(self) -> Tensor:
        """Return right shoulder, guaranteed > a via softplus offset."""
        return self.a + F.softplus(self.raw_d) + self.eps

    def forward(self, x: Tensor) -> Tensor:
        """Compute smooth S-shaped membership values for input tensor."""
        x = self._as_tensor(x)
        t = ((x - self.a) / (self.b - self.a + self.eps)).clamp(0.0, 1.0)
        return _smoothstep(t)


class LinSShapedMF(MembershipFunction):
    """Linear S-shaped membership from 0 to 1 between a and b."""

    def __init__(self, a: float, b: float, eps: float | None = None) -> None:
        """Initialize with ramp start (a) and end (b); requires a < b."""
        super().__init__(eps=eps)
        if not (a < b):
            raise ValueError("expected a < b")
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.raw_d = nn.Parameter(torch.tensor(_inv_softplus(float(b - a), eps)))

    @property
    def b(self) -> Tensor:
        """Return right shoulder, guaranteed > a via softplus offset."""
        return self.a + F.softplus(self.raw_d) + self.eps

    def forward(self, x: Tensor) -> Tensor:
        """Compute linear S-shaped membership values for input tensor."""
        x = self._as_tensor(x)
        return ((x - self.a) / (self.b - self.a + self.eps)).clamp(0.0, 1.0)


class ZShapedMF(MembershipFunction):
    """Smooth Z-shaped membership from 1 to 0 between a and b."""

    def __init__(self, a: float, b: float, eps: float | None = None) -> None:
        """Initialize with transition start (a) and end (b); requires a < b."""
        super().__init__(eps=eps)
        if not (a < b):
            raise ValueError("expected a < b")
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.raw_d = nn.Parameter(torch.tensor(_inv_softplus(float(b - a), eps)))

    @property
    def b(self) -> Tensor:
        """Return right foot, guaranteed > a via softplus offset."""
        return self.a + F.softplus(self.raw_d) + self.eps

    def forward(self, x: Tensor) -> Tensor:
        """Compute smooth Z-shaped membership values for input tensor."""
        x = self._as_tensor(x)
        t = ((x - self.a) / (self.b - self.a + self.eps)).clamp(0.0, 1.0)
        return 1.0 - _smoothstep(t)


class LinZShapedMF(MembershipFunction):
    """Linear Z-shaped membership from 1 to 0 between a and b."""

    def __init__(self, a: float, b: float, eps: float | None = None) -> None:
        """Initialize with ramp start (a) and end (b); requires a < b."""
        super().__init__(eps=eps)
        if not (a < b):
            raise ValueError("expected a < b")
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.raw_d = nn.Parameter(torch.tensor(_inv_softplus(float(b - a), eps)))

    @property
    def b(self) -> Tensor:
        """Return right foot, guaranteed > a via softplus offset."""
        return self.a + F.softplus(self.raw_d) + self.eps

    def forward(self, x: Tensor) -> Tensor:
        """Compute linear Z-shaped membership values for input tensor."""
        x = self._as_tensor(x)
        return 1.0 - ((x - self.a) / (self.b - self.a + self.eps)).clamp(0.0, 1.0)


class PiMF(MembershipFunction):
    """Pi-shaped membership with smooth S/Z transitions and flat top."""

    def __init__(self, a: float, b: float, c: float, d: float, eps: float | None = None) -> None:
        """Initialize with four boundary points; requires a < b <= c < d."""
        super().__init__(eps=eps)
        if not (a < b <= c < d):
            raise ValueError("expected a < b <= c < d")
        eps_value = self.eps
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.raw_d1 = nn.Parameter(torch.tensor(_inv_softplus(float(b - a), eps_value)))
        self.raw_d2 = nn.Parameter(torch.tensor(_inv_softplus(float(c - b + eps_value), eps_value)))
        self.raw_d3 = nn.Parameter(torch.tensor(_inv_softplus(float(d - c), eps_value)))

    def points(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return (a, b, c, d) with ordering guaranteed by softplus offsets."""
        b = self.a + F.softplus(self.raw_d1) + self.eps
        c = b + F.softplus(self.raw_d2)
        d = c + F.softplus(self.raw_d3) + self.eps
        return self.a, b, c, d

    def forward(self, x: Tensor) -> Tensor:
        """Compute Pi-shaped membership values for input tensor."""
        x = self._as_tensor(x)
        a, b, c, d = self.points()

        t_s = ((x - a) / (b - a + self.eps)).clamp(0.0, 1.0)
        s = _smoothstep(t_s)

        t_z = ((x - c) / (d - c + self.eps)).clamp(0.0, 1.0)
        z = 1.0 - _smoothstep(t_z)

        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        return torch.where(x < a, zero, torch.where(x <= b, s, torch.where(x <= c, one, torch.where(x < d, z, zero))))


class GaussianPIMF(MembershipFunction):
    """Gaussian membership with a positive infimum (PIMF).

    Defined as ``exp(-K * (1 - exp(-(x - m)^2 / (2 * sigma^2))))``.
    The infimum of this function is ``exp(-K) > 0``, which prevents
    numeric underflow when used with softmin-based T-norms on
    high-dimensional data.

    Reference: Ma et al., "An adaptive double-parameter softmin based
    Takagi-Sugeno-Kang fuzzy system for high-dimensional data", Fuzzy
    Sets and Systems 521 (2025), Eq. (41).
    """

    def __init__(
        self,
        mean: float = 0.0,
        sigma: float = 1.0,
        K: float = 1.0,
        eps: float | None = None,
    ) -> None:
        """Initialize Gaussian PIMF with center, spread, and infimum control K.

        Args:
            mean: Center of the Gaussian.
            sigma: Spread (standard deviation); must be positive.
            K: Positive infimum control parameter in (0, 745].
                The membership infimum is exp(-K).
            eps: Numerical stability constant.
        """
        super().__init__(eps=eps)
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if not (0.0 < K <= 745.0):
            raise ValueError("K must be in the interval (0, 745]")
        self.mean = nn.Parameter(torch.tensor(float(mean)))
        self.raw_sigma = nn.Parameter(torch.tensor(_inv_softplus(float(sigma), eps)))
        self.K = float(K)

    @property
    def sigma(self) -> Tensor:
        """Return the positive standard deviation via softplus reparameterization."""
        return F.softplus(self.raw_sigma) + self.eps

    def forward(self, x: Tensor) -> Tensor:
        """Compute Gaussian PIMF membership values for input tensor."""
        x = self._as_tensor(x)
        z_sq = ((x - self.mean) / self.sigma).square()
        inner = 1.0 - torch.exp(-0.5 * z_sq)
        return torch.exp(-self.K * inner)


__all__: list[str] = [
    "BellMF",
    "CompositeGaussianMF",
    "GaussianMF",
    "GaussianPIMF",
    "LinSShapedMF",
    "LinZShapedMF",
    "MembershipFunction",
    "PiMF",
    "ProdSigmoidalMF",
    "SShapedMF",
    "SigmoidalMF",
    "TrapezoidalMF",
    "TriangularMF",
    "ZShapedMF",
]
