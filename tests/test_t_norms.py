from __future__ import annotations

import pytest
import torch

from highfis.t_norms import (
    DombiTNorm,
    GMeanTNorm,
    MinimumTNorm,
    ProductTNorm,
    resolve_t_norm,
    t_norm_dombi,
    t_norm_gmean,
    t_norm_min,
    t_norm_prod,
)


def test_t_norm_prod_min_gmean_values() -> None:
    terms = torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32)

    prod = t_norm_prod(terms)
    tmin = t_norm_min(terms)
    gmean = t_norm_gmean(terms)

    assert torch.allclose(prod, torch.tensor([0.125, 0.36]), atol=1e-6)
    assert torch.allclose(tmin, torch.tensor([0.25, 0.4]), atol=1e-6)
    assert torch.allclose(gmean, torch.tensor([0.35355338, 0.6]), atol=1e-6)


def test_resolve_t_norm_returns_callable() -> None:
    assert isinstance(resolve_t_norm("prod"), ProductTNorm)
    assert isinstance(resolve_t_norm("min"), MinimumTNorm)
    assert isinstance(resolve_t_norm("gmean"), GMeanTNorm)
    assert isinstance(resolve_t_norm("dombi"), DombiTNorm)


def test_t_norm_dombi_values() -> None:
    terms = torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32)
    out = t_norm_dombi(terms, lambda_=2.0)
    expected = torch.tensor([0.2403, 0.3993], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-4)


def test_resolve_t_norm_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="t_norm must be"):
        resolve_t_norm("unknown")
