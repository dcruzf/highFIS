"""Tests for new membership functions (Triangular, Trapezoidal, Bell, Sigmoidal)."""

from __future__ import annotations

import pytest
import torch

from highfis.memberships import BellMF, SigmoidalMF, TrapezoidalMF, TriangularMF


class TestTriangularMF:
    def test_peak_at_center(self) -> None:
        mf = TriangularMF(left=-1.0, center=0.0, right=1.0)
        y = mf(torch.tensor([0.0]))
        assert torch.allclose(y, torch.tensor([1.0]), atol=1e-4)

    def test_zero_outside_support(self) -> None:
        mf = TriangularMF(left=-1.0, center=0.0, right=1.0)
        y = mf(torch.tensor([-2.0, 2.0]))
        assert torch.allclose(y, torch.zeros(2), atol=1e-4)

    def test_output_in_unit_interval(self) -> None:
        mf = TriangularMF(left=-1.0, center=0.0, right=1.0)
        x = torch.linspace(-2.0, 2.0, 50)
        y = mf(x)
        assert bool(torch.all(y >= 0.0))
        assert bool(torch.all(y <= 1.0 + 1e-5))

    def test_rejects_invalid_order(self) -> None:
        with pytest.raises(ValueError, match="must satisfy left <= center <= right"):
            TriangularMF(left=1.0, center=0.0, right=2.0)

    def test_rejects_degenerate(self) -> None:
        with pytest.raises(ValueError, match="left and right must differ"):
            TriangularMF(left=0.0, center=0.0, right=0.0)


class TestTrapezoidalMF:
    def test_flat_top(self) -> None:
        mf = TrapezoidalMF(a=-2.0, b=-1.0, c=1.0, d=2.0)
        y = mf(torch.tensor([0.0, -0.5, 0.5]))
        assert torch.allclose(y, torch.ones(3), atol=1e-4)

    def test_zero_outside_support(self) -> None:
        mf = TrapezoidalMF(a=-2.0, b=-1.0, c=1.0, d=2.0)
        y = mf(torch.tensor([-3.0, 3.0]))
        assert torch.allclose(y, torch.zeros(2), atol=1e-4)

    def test_output_in_unit_interval(self) -> None:
        mf = TrapezoidalMF(a=-2.0, b=-1.0, c=1.0, d=2.0)
        x = torch.linspace(-4.0, 4.0, 50)
        y = mf(x)
        assert bool(torch.all(y >= 0.0))
        assert bool(torch.all(y <= 1.0 + 1e-5))

    def test_rejects_invalid_order(self) -> None:
        with pytest.raises(ValueError, match="must satisfy a <= b <= c <= d"):
            TrapezoidalMF(a=1.0, b=-1.0, c=1.0, d=2.0)

    def test_rejects_degenerate(self) -> None:
        with pytest.raises(ValueError, match="a and d must differ"):
            TrapezoidalMF(a=0.0, b=0.0, c=0.0, d=0.0)


class TestBellMF:
    def test_peak_at_center(self) -> None:
        mf = BellMF(a=1.0, b=2.0, center=0.0)
        y = mf(torch.tensor([0.0]))
        assert torch.allclose(y, torch.tensor([1.0]), atol=1e-4)

    def test_output_in_unit_interval(self) -> None:
        mf = BellMF(a=1.0, b=2.0, center=0.0)
        x = torch.linspace(-5.0, 5.0, 50)
        y = mf(x)
        assert bool(torch.all(y >= 0.0))
        assert bool(torch.all(y <= 1.0 + 1e-5))

    def test_rejects_non_positive_a(self) -> None:
        with pytest.raises(ValueError, match="a must be positive"):
            BellMF(a=0.0, b=2.0, center=0.0)

    def test_rejects_non_positive_b(self) -> None:
        with pytest.raises(ValueError, match="b must be positive"):
            BellMF(a=1.0, b=0.0, center=0.0)


class TestSigmoidalMF:
    def test_half_at_center(self) -> None:
        mf = SigmoidalMF(a=10.0, center=0.0)
        y = mf(torch.tensor([0.0]))
        assert torch.allclose(y, torch.tensor([0.5]), atol=1e-4)

    def test_approaches_one_far_right(self) -> None:
        mf = SigmoidalMF(a=10.0, center=0.0)
        y = mf(torch.tensor([10.0]))
        assert float(y.item()) > 0.99

    def test_approaches_zero_far_left(self) -> None:
        mf = SigmoidalMF(a=10.0, center=0.0)
        y = mf(torch.tensor([-10.0]))
        assert float(y.item()) < 0.01

    def test_output_in_unit_interval(self) -> None:
        mf = SigmoidalMF(a=1.0, center=0.0)
        x = torch.linspace(-5.0, 5.0, 50)
        y = mf(x)
        assert bool(torch.all(y >= 0.0))
        assert bool(torch.all(y <= 1.0 + 1e-5))
