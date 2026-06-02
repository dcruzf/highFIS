from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from highfis.memberships import (
    BellMF,
    CompositeExponentialMF,
    CompositeGaussianMF,
    CompositeGMF,
    ConstantMF,
    DiffSigmoidalMF,
    DimensionDependentGaussianMF,
    GaussianMF,
    GaussianPiMF,
    LinSShapedMF,
    LinZShapedMF,
    MembershipFunction,
    PiMF,
    ProdSigmoidalMF,
    SigmoidalMF,
    SShapedMF,
    TrapezoidalMF,
    TriangularMF,
    ZShapedMF,
)
from highfis.models import TSKRegressorModel


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


def test_dimension_dependent_gaussian_mf_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dimension must be greater than 1"):
        DimensionDependentGaussianMF(mean=0.0, sigma=1.0, dimension=1)


def test_dimension_dependent_gaussian_mf_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        DimensionDependentGaussianMF(mean=0.0, sigma=0.0, dimension=1000, xi=745.0)


def test_dimension_dependent_gaussian_mf_rejects_invalid_xi() -> None:
    with pytest.raises(ValueError, match="xi must be greater than 1"):
        DimensionDependentGaussianMF(mean=0.0, sigma=1.0, dimension=1000, xi=1.0)


def test_dimension_dependent_gaussian_mf_forward_peaks_at_mean() -> None:
    mf = DimensionDependentGaussianMF(mean=0.0, sigma=1.0, dimension=1000, xi=745.0)
    x = torch.tensor([0.0], dtype=torch.float32)
    y = mf(x)
    assert y.shape == (1,)
    assert torch.allclose(y, torch.tensor([1.0]), atol=1e-5)


def test_dimension_dependent_gaussian_mf_outputs_in_unit_interval() -> None:
    mf = DimensionDependentGaussianMF(mean=0.0, sigma=1.0, dimension=1000, xi=745.0)
    x = torch.linspace(-3.0, 3.0, 11)
    y = mf(x)
    assert y.shape == x.shape
    assert bool(torch.all(y >= 0.0))
    assert bool(torch.all(y <= 1.0))


def test_composite_exponential_mf_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        CompositeExponentialMF(center=0.0, sigma=0.0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_composite_gmf_forward_lower_bound() -> None:
    mf = CompositeGMF(mean=0.0, sigma=1.0)
    x = torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float32)
    y = mf(x)
    assert y.shape == x.shape
    assert bool(torch.all(y > 0.0))
    assert bool(torch.all(y <= 1.0))
    assert float(torch.min(y).item()) == pytest.approx(math.exp(-1.0), rel=1e-6)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_composite_gmf_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        CompositeGMF(mean=0.0, sigma=0.0)


def test_composite_gmf_emits_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="CompositeGMF is deprecated"):
        CompositeGMF(mean=0.0, sigma=1.0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_composite_gmf_inspect_params_returns_mean_sigma() -> None:
    mf = CompositeGMF(mean=0.5, sigma=2.0)
    params = mf.inspect_params()
    assert set(params) == {"mean", "sigma"}
    assert params["mean"] == pytest.approx(0.5)
    assert params["sigma"] == pytest.approx(2.0)


def test_composite_exponential_mf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be greater than 1"):
        CompositeExponentialMF(center=0.0, sigma=1.0, k=1.0)


def test_composite_exponential_mf_forward_peaks_at_center() -> None:
    mf = CompositeExponentialMF(center=0.5, sigma=1.0, k=10.0)
    x = torch.tensor([0.5], dtype=torch.float32)
    y = mf(x)
    assert y.shape == (1,)
    assert torch.allclose(y, torch.tensor([1.0]), atol=1e-5)


def test_composite_exponential_mf_has_lower_bound() -> None:
    mf = CompositeExponentialMF(center=0.0, sigma=1.0, k=10.0)
    x = torch.tensor([10.0], dtype=torch.float32)
    y = mf(x)
    assert torch.allclose(y, torch.tensor([0.1]), atol=1e-4)


def test_composite_exponential_mf_outputs_in_unit_interval() -> None:
    mf = CompositeExponentialMF(center=0.0, sigma=1.0, k=10.0)
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


def test_composite_gaussian_mf_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        CompositeGaussianMF(mean=0.0, sigma=0.0)


def test_s_shaped_mf_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="expected a < b"):
        SShapedMF(a=1.0, b=0.0)


def test_lin_s_shaped_mf_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="expected a < b"):
        LinSShapedMF(a=1.0, b=0.0)


def test_z_shaped_mf_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="expected a < b"):
        ZShapedMF(a=1.0, b=0.0)


def test_lin_z_shaped_mf_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="expected a < b"):
        LinZShapedMF(a=1.0, b=0.0)


def test_pi_mf_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="expected a < b <= c < d"):
        PiMF(a=0.0, b=1.0, c=0.5, d=2.0)


def test_gaussian_pimf_rejects_invalid_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        GaussianPiMF(mean=0.0, sigma=0.0, k=1.0)


def test_gaussian_pimf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match=r"k must be in the interval \(0, 745\]"):
        GaussianPiMF(mean=0.0, sigma=1.0, k=0.0)


def test_gaussian_pimf_infimum_positive() -> None:
    mf = GaussianPiMF(mean=0.0, sigma=1.0, k=2.0)
    x = torch.tensor([100.0])
    y = mf(x)
    assert y.shape == x.shape
    assert bool(torch.all(y > 0.0))


@pytest.mark.parametrize(
    "mf, expected",
    [
        (ConstantMF(value=0.5), {"value": 0.5}),
        (GaussianMF(mean=0.5, sigma=1.5), {"mean": 0.5, "sigma": 1.5}),
        (
            DimensionDependentGaussianMF(mean=0.5, sigma=2.0, dimension=1000, xi=745.0),
            {
                "mean": 0.5,
                "sigma": 2.0,
                "dimension": 1000.0,
                "xi": 745.0,
                "rho": float(1.0 - math.log(745.0) / math.log(1000.0)),
                "paper_strict_equation": False,
            },
        ),
        (CompositeGaussianMF(mean=0.5, sigma=2.0), {"mean": 0.5, "sigma": 2.0}),
        (TriangularMF(left=-1.0, center=0.0, right=1.0), {"left": -1.0, "center": 0.0, "right": 1.0}),
        (TrapezoidalMF(a=0.0, b=1.0, c=2.0, d=3.0), {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}),
        (BellMF(a=1.0, b=2.0, center=0.0), {"a": 1.0, "b": 2.0, "center": 0.0}),
        (CompositeExponentialMF(center=0.5, sigma=2.0, k=10.0), {"center": 0.5, "sigma": 2.0, "k": 10.0}),
        (SigmoidalMF(a=1.0, center=0.0), {"a": 1.0, "center": 0.0}),
        (
            DiffSigmoidalMF(a1=2.0, center1=-1.0, a2=3.0, center2=1.0),
            {"a1": 2.0, "center1": -1.0, "a2": 3.0, "center2": 1.0},
        ),
        (
            ProdSigmoidalMF(a1=2.0, center1=-1.0, a2=-3.0, center2=1.0),
            {"a1": 2.0, "center1": -1.0, "a2": -3.0, "center2": 1.0},
        ),
        (SShapedMF(a=0.0, b=1.0), {"a": 0.0, "b": 1.0}),
        (LinSShapedMF(a=0.0, b=1.0), {"a": 0.0, "b": 1.0}),
        (ZShapedMF(a=0.0, b=1.0), {"a": 0.0, "b": 1.0}),
        (LinZShapedMF(a=0.0, b=1.0), {"a": 0.0, "b": 1.0}),
        (PiMF(a=0.0, b=0.5, c=1.0, d=1.5), {"a": 0.0, "b": 0.5, "c": 1.0, "d": 1.5}),
        (GaussianPiMF(mean=0.5, sigma=2.0, k=5.0), {"mean": 0.5, "sigma": 2.0, "k": 5.0}),
    ],
)
def test_all_membership_functions_expose_inspect_params(mf: Any, expected: dict[str, float]) -> None:
    params = mf.inspect_params()
    assert isinstance(params, dict)
    assert set(params) == set(expected)
    for key, value in expected.items():
        assert params[key] == pytest.approx(value)


def test_membership_function_default_inspect_params_with_base_class() -> None:
    class DummyMF(MembershipFunction):
        def __init__(self) -> None:
            super().__init__()
            self.raw_value = torch.nn.Parameter(torch.tensor(1.2))

        @property
        def value(self) -> float:
            return float(self.raw_value.detach())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    mf = DummyMF()
    params = mf.inspect_params()
    assert set(params) == {"value"}
    assert params["value"] == pytest.approx(1.2)


def test_tsk_regressor_get_mf_params_returns_sane_structure() -> None:
    input_mfs = {"x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.5)]}
    model = TSKRegressorModel(input_mfs)
    params = model.get_mf_params()

    assert set(params) == {"x1"}
    assert isinstance(params["x1"], list)
    assert len(params["x1"]) == 2
    assert params["x1"][0]["type"] == "GaussianMF"
    assert params["x1"][0]["mean"] == pytest.approx(0.0)
    assert params["x1"][0]["sigma"] == pytest.approx(1.0)
