from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

TNormFn = Callable[[Tensor], Tensor]


def t_norm_prod(terms: Tensor) -> Tensor:
    """Product t-norm over antecedent terms with shape (batch, n_inputs)."""
    return torch.prod(terms, dim=1)


def t_norm_min(terms: Tensor) -> Tensor:
    """Minimum t-norm over antecedent terms with shape (batch, n_inputs)."""
    return torch.min(terms, dim=1).values


def t_norm_gmean(terms: Tensor, eps: float | None = None) -> Tensor:
    """Geometric mean aggregation for HTSK defuzzification."""
    eps = torch.finfo(terms.dtype).eps if eps is None else eps
    ln_terms = terms.clamp(min=eps).log()
    return (ln_terms.mean(dim=1)).exp()


def t_norm_dombi(terms: Tensor, lambda_: float = 1.0, eps: float | None = None) -> Tensor:
    """Dombi t-norm aggregation over antecedent terms."""
    if lambda_ <= 0.0:
        raise ValueError("lambda_ must be > 0")

    eps = torch.finfo(terms.dtype).eps if eps is None else eps
    clamped = terms.clamp(min=eps, max=1.0)
    inv = (1.0 / clamped) - 1.0
    powered = torch.pow(inv, float(lambda_))
    summed = powered.sum(dim=1)
    return 1.0 / (1.0 + torch.pow(summed, 1.0 / float(lambda_)))


def resolve_t_norm(name: str) -> TNormFn:
    """Resolve a built-in t-norm by name."""
    if name == "prod":
        return t_norm_prod
    if name == "min":
        return t_norm_min
    if name == "gmean":
        return t_norm_gmean
    if name == "dombi":
        return t_norm_dombi
    raise ValueError("t_norm must be 'prod', 'min', 'gmean', or 'dombi'")


__all__: list[str] = [
    "TNormFn",
    "resolve_t_norm",
    "t_norm_dombi",
    "t_norm_gmean",
    "t_norm_min",
    "t_norm_prod",
]
