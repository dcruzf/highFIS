from __future__ import annotations

import pytest
import torch

from highfis.memberships import GaussianMF


def test_gaussian_mf_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        GaussianMF(mean=0.0, sigma=0.0)


def test_gaussian_mf_forward_peaks_at_mean() -> None:
    mf = GaussianMF(mean=0.5, sigma=1.0)
    x = torch.tensor([0.5], dtype=torch.float32)
    y = mf(x)
    assert y.shape == (1,)
    assert torch.allclose(y, torch.tensor([1.0]), atol=1e-5)


def test_gaussian_mf_outputs_in_unit_interval() -> None:
    mf = GaussianMF(mean=0.0, sigma=1.0)
    x = torch.linspace(-3.0, 3.0, 11)
    y = mf(x)
    assert y.shape == x.shape
    assert bool(torch.all(y >= 0.0))
    assert bool(torch.all(y <= 1.0))
