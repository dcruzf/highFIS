from __future__ import annotations

import math

import pytest
import torch

from highfis.t_norms import (
    AdaptiveDombiTNorm,
    ALESoftminYagerTNorm,
    DombiTNorm,
    GMeanTNorm,
    MinimumTNorm,
    ProductTNorm,
    YagerSimpleTNorm,
    YagerTNorm,
    resolve_t_norm,
)


def test_t_norm_prod_min_gmean_values() -> None:
    terms = torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32)

    prod = ProductTNorm()(terms, dim=1)
    tmin = MinimumTNorm()(terms, dim=1)
    gmean = GMeanTNorm()(terms, dim=1)

    assert torch.allclose(prod, torch.tensor([0.125, 0.36]), atol=1e-6)
    assert torch.allclose(tmin, torch.tensor([0.25, 0.4]), atol=1e-6)
    assert torch.allclose(gmean, torch.tensor([0.35355338, 0.6]), atol=1e-6)


def test_resolve_t_norm_returns_callable() -> None:
    assert isinstance(resolve_t_norm("prod"), ProductTNorm)
    assert isinstance(resolve_t_norm("min"), MinimumTNorm)
    assert isinstance(resolve_t_norm("gmean"), GMeanTNorm)
    assert isinstance(resolve_t_norm("dombi"), DombiTNorm)
    assert isinstance(resolve_t_norm("yager"), YagerTNorm)
    assert isinstance(resolve_t_norm("yager_simple"), YagerSimpleTNorm)
    assert isinstance(resolve_t_norm("ale_softmin_yager"), ALESoftminYagerTNorm)
    assert isinstance(resolve_t_norm("ale-yager"), ALESoftminYagerTNorm)
    assert isinstance(resolve_t_norm("yager_ale"), ALESoftminYagerTNorm)


def test_t_norm_dombi_values() -> None:
    terms = torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32)
    out = DombiTNorm(lambda_=2.0)(terms, dim=1)
    expected = torch.tensor([0.2403, 0.3993], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-4)


def test_t_norm_yager_values() -> None:
    terms = torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32)
    out = YagerTNorm(lambda_=2.0)(terms, dim=1)
    expected = 1.0 - torch.minimum((1.0 - terms).pow(2.0).sum(dim=1).pow(0.5), torch.tensor(1.0))
    assert torch.allclose(out, expected, atol=1e-6)


def test_t_norm_yager_simple_values() -> None:
    terms = torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32)
    out = YagerSimpleTNorm(lambda_=2.0)(terms, dim=1)
    expected = 1.0 - (1.0 - terms).pow(2.0).sum(dim=1).pow(0.5)
    assert torch.allclose(out, expected, atol=1e-6)


def test_yager_tnorm_forward() -> None:
    terms = torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32)
    out_cls = YagerTNorm(lambda_=2.0)(terms, dim=1)
    expected = 1.0 - torch.minimum((1.0 - terms).pow(2.0).sum(dim=1).pow(0.5), torch.tensor(1.0))
    assert torch.allclose(out_cls, expected, atol=1e-6)


def test_yager_simple_tnorm_class_matches_function() -> None:
    out_cls = YagerSimpleTNorm(lambda_=2.0)(torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32), dim=1)
    assert out_cls.shape == torch.Size([2])


def test_ale_softmin_yager_tnorm_class_matches_function() -> None:
    out_cls = ALESoftminYagerTNorm(lambda_=2.0)(torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32), dim=1)
    assert out_cls.shape == torch.Size([2])


def test_yager_tnorm_clips_to_zero() -> None:
    terms = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    out = YagerTNorm(lambda_=0.5)(terms, dim=1)
    assert out.shape == torch.Size([1])
    assert out[0].item() == pytest.approx(0.0)


def test_yager_simple_tnorm_can_be_negative() -> None:
    terms = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    out = YagerSimpleTNorm(lambda_=0.5)(terms, dim=1)
    assert out[0].item() < 0.0


def test_t_norm_ale_softmin_yager_values() -> None:
    out = ALESoftminYagerTNorm(lambda_=2.0)(torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32), dim=1)
    assert out.shape == torch.Size([2])
    assert bool(torch.all(out >= 0.0))
    assert bool(torch.all(out <= 1.0))


def test_resolve_t_norm_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="t_norm must be"):
        resolve_t_norm("unknown")


def test_yager_tnorm_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        YagerTNorm(lambda_=0.0)


def test_yager_simple_tnorm_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        YagerSimpleTNorm(lambda_=0.0)


def test_ale_softmin_yager_tnorm_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        ALESoftminYagerTNorm(lambda_=0.0)


def test_dombi_tnorm_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        DombiTNorm(lambda_=0.0)


def test_adaptive_dombi_tnorm_values() -> None:
    norm = AdaptiveDombiTNorm(dimension=1000, lower_bound=1.0 / math.e, K=10.0)
    terms = torch.tensor([[0.25, 0.5], [0.4, 0.9]], dtype=torch.float32)
    out = norm(terms, dim=1)
    assert out.shape == torch.Size([2])
    assert bool(torch.all(out >= 0.0))
    assert bool(torch.all(out <= 1.0))


def test_adaptive_dombi_tnorm_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="dimension must be > 1"):
        AdaptiveDombiTNorm(dimension=1, lower_bound=0.1)
    with pytest.raises(ValueError, match=r"lower_bound must be in \[0, 1\)"):
        AdaptiveDombiTNorm(dimension=1000, lower_bound=1.0)
    with pytest.raises(ValueError, match="K must be > 1"):
        AdaptiveDombiTNorm(dimension=1000, lower_bound=0.1, K=1.0)


def test_resolve_adaptive_dombi_t_norm_rejects_name() -> None:
    with pytest.raises(ValueError, match="adaptive_dombi requires a dimension"):
        resolve_t_norm("adaptive_dombi")
