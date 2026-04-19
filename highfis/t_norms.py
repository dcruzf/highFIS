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


def t_norm_prod(terms: Tensor) -> Tensor:
    """Product t-norm over antecedent terms with shape (batch, n_inputs)."""
    return ProductTNorm()(terms, dim=1)


def t_norm_min(terms: Tensor) -> Tensor:
    """Minimum t-norm over antecedent terms with shape (batch, n_inputs)."""
    return MinimumTNorm()(terms, dim=1)


def t_norm_gmean(terms: Tensor, eps: float | None = None) -> Tensor:
    """Geometric mean aggregation for HTSK defuzzification."""
    return GMeanTNorm(eps=eps)(terms, dim=1)


def t_norm_dombi(terms: Tensor, lambda_: float = 1.0, eps: float | None = None) -> Tensor:
    """Dombi t-norm aggregation over antecedent terms."""
    return DombiTNorm(lambda_=lambda_, eps=eps)(terms, dim=1)


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
    raise ValueError("t_norm must be 'prod', 'min', 'gmean', or 'dombi'")


__all__: list[str] = [
    "BaseTNorm",
    "DombiTNorm",
    "GMeanTNorm",
    "MinimumTNorm",
    "ProductTNorm",
    "TNormFn",
    "resolve_t_norm",
    "t_norm_dombi",
    "t_norm_gmean",
    "t_norm_min",
    "t_norm_prod",
]
