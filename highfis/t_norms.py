"""Fuzzy T-norm aggregation strategies for TSK antecedent computation.

This module defines learnable T-norm strategies used by ``highfis.layers``
to aggregate per-input membership degrees into rule firing strengths.
Each strategy is implemented as a subclass of ``BaseTNorm``.

Built-in T-norm classes:
    - ``ProductTNorm`` — standard product conjunction.
    - ``MinimumTNorm`` — Gödel / minimum conjunction.
    - ``GMeanTNorm`` — geometric mean, the default for HTSK.
    - ``DombiTNorm`` — Dombi parametric T-norm (``lambda_ > 0``).
    - ``AdaptiveDombiTNorm`` — Dombi T-norm with adaptive lambda selection.
    - ``YagerTNorm`` — Yager parametric T-norm (``lambda_ > 0``).
    - ``YagerSimpleTNorm`` — simplified Yager without the outer minimum.
    - ``ALESoftminYagerTNorm`` — ALE-softmin Yager variant.

Helper functions:
    - ``resolve_t_norm(name)`` — map string names to T-norm instances.
      Supported names include ``"prod"``, ``"min"``, ``"gmean"``,
      ``"dombi"``, ``"yager"``, ``"yager_simple"``, and
      ``"ale_softmin_yager"``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

TNormFn = Callable[..., Tensor]


class BaseTNorm(nn.Module, ABC):
    """Base class for T-norm strategies."""

    @abstractmethod
    def forward(self, terms: Tensor, dim: int = -1) -> Tensor:
        """Apply the T-norm aggregation over the specified dimension.

        Args:
            terms: Tensor of shape ``(batch, n_rules, n_inputs)``
                containing per-rule, per-input membership degrees.
            dim: Dimension over which to aggregate.  Defaults to
                ``-1`` (the input dimension).

        Returns:
            Tensor of shape ``(batch, n_rules)`` with aggregated
            firing strengths.
        """
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
        """Initialize the geometric mean T-norm.

        Args:
            eps: Small positive constant for clamping inputs before
                computing the log.  ``None`` infers it from
                :func:`torch.finfo` for the input dtype.
        """
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
        r"""Initialize the Dombi T-norm.

        Args:
            lambda_: Positive shape parameter $\lambda > 0$.  Higher
                values make the T-norm approach the minimum; lower
                values make it approach the product.
            eps: Small positive constant for clamping inputs.
                ``None`` infers it from :func:`torch.finfo` for the
                input dtype.

        Raises:
            ValueError: If *lambda_* is not positive.
        """
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


class AdaptiveDombiTNorm(BaseTNorm):
    """Adaptive Dombi T-norm strategy with automatically selected lambda."""

    def __init__(
        self,
        dimension: int,
        lower_bound: float = 1.0 / math.e,
        k: float = 10.0,
        eps: float | None = None,
    ) -> None:
        r"""Initialize the adaptive Dombi T-norm.

        Args:
            dimension: Number of input features D.
            lower_bound: Positive lower bound of the membership function.
            k: Heuristic scaling constant used to compute lambda.
            eps: Small positive constant for clamping inputs.

        Raises:
            ValueError: If arguments are invalid or lambda cannot be computed.
        """
        super().__init__()
        if dimension <= 1:
            raise ValueError("dimension must be > 1")
        if not 0.0 <= lower_bound < 1.0:
            raise ValueError("lower_bound must be in [0, 1)")
        if k <= 1.0:
            raise ValueError("k must be > 1")

        self.dimension = int(dimension)
        self.lower_bound = float(lower_bound)
        self.k = float(k)
        self.eps = eps

        denom = math.log(self.k - self.lower_bound) - math.log(1.0 - self.lower_bound)
        if denom <= 0.0:  # pragma: no cover
            raise ValueError("invalid lambda computation for given lower_bound and K")
        lambda_ = math.log(float(self.dimension)) / denom
        self.dombi = DombiTNorm(lambda_=lambda_, eps=eps)

    def forward(self, terms: Tensor, dim: int = -1) -> Tensor:
        """Compute the Dombi aggregation with the adaptive lambda parameter."""
        return self.dombi(terms, dim=dim)


class YagerTNorm(BaseTNorm):
    """Yager T-norm strategy."""

    def __init__(self, lambda_: float = 1.0, eps: float | None = None) -> None:
        r"""Initialize the Yager T-norm.

        Args:
            lambda_: Positive shape parameter $\lambda > 0$.  Higher
                values make the T-norm approach the minimum; lower
                values make it approach the product.
            eps: Small positive constant for clamping inputs.
                ``None`` infers it from :func:`torch.finfo` for the
                input dtype.

        Raises:
            ValueError: If *lambda_* is not positive.
        """
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


def resolve_t_norm(name: str | TNormFn) -> TNormFn:
    """Resolve a t-norm by name or return a callable directly."""
    if callable(name):
        return cast(TNormFn, name)
    if name == "prod":
        return ProductTNorm()
    if name == "min":
        return MinimumTNorm()
    if name == "gmean":
        return GMeanTNorm()
    if name == "dombi":
        return DombiTNorm()
    if name == "adaptive_dombi":
        raise ValueError("adaptive_dombi requires a dimension and lower_bound; instantiate AdaptiveDombiTNorm directly")
    if name == "yager":
        return YagerTNorm()
    if name == "yager_simple":
        return YagerSimpleTNorm()
    if name in {"ale_softmin_yager", "ale-yager", "yager_ale"}:
        return ALESoftminYagerTNorm()
    raise ValueError("t_norm must be 'prod', 'min', 'gmean', 'dombi', 'yager', 'yager_simple', or 'ale_softmin_yager'")


__all__: list[str] = [
    "ALESoftminYagerTNorm",
    "AdaptiveDombiTNorm",
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
