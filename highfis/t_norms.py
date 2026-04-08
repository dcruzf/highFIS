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


def t_norm_gmean(terms: Tensor) -> Tensor:
    """Geometric mean aggregation for HTSK defuzzification."""
    eps = 1e-30
    ln_terms = terms.clamp(min=eps).log()
    return (ln_terms.mean(dim=1)).exp()


def resolve_t_norm(name: str) -> TNormFn:
    """Resolve a built-in t-norm by name."""
    if name == "prod":
        return t_norm_prod
    if name == "min":
        return t_norm_min
    if name == "gmean":
        return t_norm_gmean
    raise ValueError("t_norm must be 'prod', 'min', or 'gmean'")


__all__ = ["TNormFn", "t_norm_prod", "t_norm_min", "t_norm_gmean", "resolve_t_norm"]
