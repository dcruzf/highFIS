"""Definitions and utilities for fuzzy T-norm aggregation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import Tensor, nn

TNormFn = Callable[..., Tensor]


class BaseTNorm(nn.Module, ABC):
    """Base class for T-norm strategies."""

    @abstractmethod
    def forward(self, terms: Tensor, dim: int = -1) -> Tensor:
        """Apply the T-norm aggregation over the specified dimension."""
        ...


class ProductTNorm(BaseTNorm):
    """Product T-norm."""

    def forward(self, terms: Tensor, dim: int = -1) -> Tensor:
        """Compute the product over the specified dimension."""
        return torch.prod(terms, dim=dim)


class MinimumTNorm(BaseTNorm):
    """Minimum T-norm."""

    def forward(self, terms: Tensor, dim: int = -1) -> Tensor:
        """Compute the minimum over the specified dimension."""
        return torch.min(terms, dim=dim).values


class GMeanTNorm(BaseTNorm):
    """Geometric mean T-norm."""

    def __init__(self, eps: float | None = None) -> None:
        """Initialize the geometric mean t-norm with an optional epsilon."""
        super().__init__()
        self.eps = eps

    def forward(self, terms: Tensor, dim: int = -1) -> Tensor:
        """Compute the geometric mean over the specified dimension."""
        eps = torch.finfo(terms.dtype).eps if self.eps is None else self.eps
        ln_terms = terms.clamp(min=eps).log()
        return ln_terms.mean(dim=dim).exp()


class DombiTNorm(BaseTNorm):
    """Dombi T-norm strategy."""

    def __init__(self, lambda_: float = 1.0, eps: float | None = None) -> None:
        """Initialize the Dombi t-norm with a lambda parameter and optional epsilon."""
        super().__init__()
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")
        self.lambda_ = float(lambda_)
        self.eps = eps

    def forward(self, terms: Tensor, dim: int = -1) -> Tensor:
        """Compute the Dombi aggregation over the specified dimension."""
        eps = torch.finfo(terms.dtype).eps if self.eps is None else self.eps
        clamped = terms.clamp(min=eps, max=1.0)
        inv = (1.0 / clamped) - 1.0
        powered = torch.pow(inv, self.lambda_)
        summed = powered.sum(dim=dim)
        return 1.0 / (1.0 + torch.pow(summed, 1.0 / self.lambda_))


class YagerTNorm(BaseTNorm):
    """Yager T-norm strategy."""

    def __init__(self, lambda_: float = 1.0, eps: float | None = None) -> None:
        """Initialize the Yager t-norm with a lambda parameter and optional epsilon."""
        super().__init__()
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")
        self.lambda_ = float(lambda_)
        self.eps = eps

    def forward(self, terms: Tensor, dim: int = -1) -> Tensor:
        """Compute the Yager aggregation over the specified dimension."""
        eps = torch.finfo(terms.dtype).eps if self.eps is None else self.eps
        clamped = terms.clamp(min=eps, max=1.0)
        power_sum = (1.0 - clamped).pow(self.lambda_).sum(dim=dim)
        result = 1.0 - power_sum.pow(1.0 / self.lambda_)
        return torch.maximum(result, torch.tensor(0.0, dtype=terms.dtype, device=terms.device))


class YagerSimpleTNorm(BaseTNorm):
    """Simplified Yager T-norm strategy without an extra minimum operator."""

    def __init__(self, lambda_: float = 1.0, eps: float | None = None) -> None:
        """Initialize the simplified Yager t-norm with a lambda parameter and optional epsilon."""
        super().__init__()
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")
        self.lambda_ = float(lambda_)
        self.eps = eps

    def forward(self, terms: Tensor, dim: int = -1) -> Tensor:
        """Compute the simplified Yager aggregation over the specified dimension."""
        eps = torch.finfo(terms.dtype).eps if self.eps is None else self.eps
        clamped = terms.clamp(min=eps, max=1.0)
        power_sum = (1.0 - clamped).pow(self.lambda_).sum(dim=dim)
        return 1.0 - power_sum.pow(1.0 / self.lambda_)


class ALESoftminYagerTNorm(BaseTNorm):
    """ALE-softmin based Yager T-norm strategy."""

    def __init__(self, lambda_: float = 1.0, eps: float | None = None) -> None:
        """Initialize the ALE-softmin Yager t-norm with a lambda parameter and optional epsilon."""
        super().__init__()
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")
        self.lambda_ = float(lambda_)
        self.eps = eps

    @staticmethod
    def _adaptive_softmin(values: Tensor, dim: int = -1) -> Tensor:
        values = values.double()
        q = -700.0 / values.data.max(dim=dim).values
        return (values * q.unsqueeze(dim=dim)).exp().sum(dim=dim).log() / q

    def forward(self, terms: Tensor, dim: int = -1) -> Tensor:
        """Compute ALE-softmin Yager aggregation over the specified dimension."""
        eps = torch.finfo(terms.dtype).eps if self.eps is None else self.eps
        clamped = terms.clamp(min=eps, max=1.0)
        y = (1.0 - clamped).pow(self.lambda_).sum(dim=dim).pow(1.0 / self.lambda_)
        softmin = self._adaptive_softmin(torch.stack([torch.ones_like(y), y], dim=0), dim=0)
        return 1.0 - softmin


def resolve_t_norm(name: str) -> TNormFn:
    """Resolve a built-in t-norm by name."""
    if name == "prod":
        return ProductTNorm()
    if name == "min":
        return MinimumTNorm()
    if name == "gmean":
        return GMeanTNorm()
    if name == "dombi":
        return DombiTNorm()
    if name == "yager":
        return YagerTNorm()
    if name == "yager_simple":
        return YagerSimpleTNorm()
    if name in {"ale_softmin_yager", "ale-yager", "yager_ale"}:
        return ALESoftminYagerTNorm()
    raise ValueError("t_norm must be 'prod', 'min', 'gmean', 'dombi', 'yager', 'yager_simple', or 'ale_softmin_yager'")


__all__: list[str] = [
    "ALESoftminYagerTNorm",
    "BaseTNorm",
    "DombiTNorm",
    "GMeanTNorm",
    "MinimumTNorm",
    "ProductTNorm",
    "TNormFn",
    "YagerSimpleTNorm",
    "YagerTNorm",
    "resolve_t_norm",
]
