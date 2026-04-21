from __future__ import annotations

import pytest
import torch

from highfis.memberships import (
    DiffSigmoidalMF,
    GaussianMF,
    GaussianPIMF,
    LinSShapedMF,
    LinZShapedMF,
    PiMF,
    ProdSigmoidalMF,
    SigmoidalMF,
    SShapedMF,
    ZShapedMF,
)


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


def test_gaussian_mf_as_tensor_accepts_float() -> None:
    """_as_tensor with a float (not Tensor) covers the float branch (line 27)."""
    mf = GaussianMF(mean=0.0, sigma=1.0)
    result = mf._as_tensor(1.5)
    assert isinstance(result, torch.Tensor)
    assert float(result) == pytest.approx(1.5)


def test_membership_function_uses_default_dtype_eps() -> None:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        mf = GaussianMF(mean=0.0, sigma=1.0)
        assert mf.eps == torch.finfo(torch.float16).eps
    finally:
        torch.set_default_dtype(previous_dtype)


def test_sigmoidal_mf_outputs_between_zero_and_one() -> None:
    mf = SigmoidalMF(a=1.0, center=0.0)
    x = torch.linspace(-3.0, 3.0, 7)
    y = mf(x)
    assert bool(torch.all(y >= 0.0))
    assert bool(torch.all(y <= 1.0))


def test_diff_sigmoidal_mf_forms_window() -> None:
    mf = DiffSigmoidalMF(a1=10.0, center1=-1.0, a2=10.0, center2=1.0)
    x = torch.tensor([-2.0, 0.0, 2.0])
    y = mf(x)
    assert y.shape == x.shape
    assert y[1] > y[0]
    assert y[1] > y[2]


def test_prod_sigmoidal_mf_forms_window() -> None:
    mf = ProdSigmoidalMF(a1=10.0, center1=-1.0, a2=-10.0, center2=1.0)
    x = torch.tensor([-2.0, 0.0, 2.0])
    y = mf(x)
    assert y.shape == x.shape
    assert y[1] > y[0]
    assert y[1] > y[2]


def test_s_shaped_mf_transitions_smoothly() -> None:
    mf = SShapedMF(a=0.0, b=1.0)
    x = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
    y = mf(x)
    assert torch.allclose(y[0], torch.tensor(0.0))
    assert torch.allclose(y[-2], torch.tensor(1.0), atol=1e-5)
    assert bool(torch.all(y >= 0.0))
    assert bool(torch.all(y <= 1.0))


def test_lin_s_shaped_mf_transitions_linearly() -> None:
    mf = LinSShapedMF(a=0.0, b=1.0)
    x = torch.tensor([0.0, 0.5, 1.0])
    y = mf(x)
    assert torch.allclose(y, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)


def test_z_shaped_mf_transitions_smoothly() -> None:
    mf = ZShapedMF(a=0.0, b=1.0)
    x = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
    y = mf(x)
    assert torch.allclose(y[0], torch.tensor(1.0))
    assert torch.allclose(y[-2], torch.tensor(0.0), atol=1e-5)
    assert bool(torch.all(y >= 0.0))
    assert bool(torch.all(y <= 1.0))


def test_lin_z_shaped_mf_transitions_linearly() -> None:
    mf = LinZShapedMF(a=0.0, b=1.0)
    x = torch.tensor([0.0, 0.5, 1.0])
    y = mf(x)
    assert torch.allclose(y, torch.tensor([1.0, 0.5, 0.0]), atol=1e-5)


def test_pi_mf_has_flat_top() -> None:
    mf = PiMF(a=0.0, b=0.5, c=1.0, d=1.5)
    x = torch.tensor([0.25, 0.75, 1.25])
    y = mf(x)
    assert y[0] < y[1]
    assert float(y[1].detach()) == pytest.approx(1.0, rel=1e-5)
    assert y[1] > y[2]
    assert bool(torch.all(y >= 0.0))
    assert bool(torch.all(y <= 1.0))


def test_gaussian_pimf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match=r"K must be in the interval \(0, 745\]"):
        GaussianPIMF(mean=0.0, sigma=1.0, K=0.0)


def test_gaussian_pimf_infimum_positive() -> None:
    mf = GaussianPIMF(mean=0.0, sigma=1.0, K=2.0)
    x = torch.tensor([100.0])
    y = mf(x)
    assert y.item() > 0.0
    assert y.item() < 1.0
