"""Tests for highfis.defuzzifiers."""

from __future__ import annotations

import pytest
import torch

from highfis.defuzzifiers import InvLogDefuzzifier, LogSumDefuzzifier, SoftmaxLogDefuzzifier, SumBasedDefuzzifier


@pytest.fixture
def firing_strengths() -> torch.Tensor:
    return torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]], dtype=torch.float32)


class TestSoftmaxLogDefuzzifier:
    def test_output_sums_to_one(self, firing_strengths: torch.Tensor) -> None:
        d = SoftmaxLogDefuzzifier()
        out = d(firing_strengths)
        assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_output_shape(self, firing_strengths: torch.Tensor) -> None:
        d = SoftmaxLogDefuzzifier()
        assert d(firing_strengths).shape == firing_strengths.shape

    def test_rejects_1d_input(self) -> None:
        d = SoftmaxLogDefuzzifier()
        with pytest.raises(ValueError, match="expected w with 2 dims"):
            d(torch.tensor([0.5, 0.5]))

    def test_dynamic_eps_uses_input_dtype(self) -> None:
        w = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float16)
        d = SoftmaxLogDefuzzifier()
        out = d(w)
        assert not torch.isnan(out).any()
        assert torch.allclose(out.sum(dim=1), torch.ones(2, dtype=torch.float16), atol=1e-3)


class TestSumBasedDefuzzifier:
    def test_output_sums_to_one(self, firing_strengths: torch.Tensor) -> None:
        d = SumBasedDefuzzifier()
        out = d(firing_strengths)
        assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_preserves_relative_ordering(self, firing_strengths: torch.Tensor) -> None:
        d = SumBasedDefuzzifier()
        out = d(firing_strengths)
        # Within each row, the ordering should be preserved
        assert bool(out[0, 1] > out[0, 0])
        assert bool(out[0, 1] > out[0, 2])

    def test_rejects_1d_input(self) -> None:
        d = SumBasedDefuzzifier()
        with pytest.raises(ValueError, match="expected w with 2 dims"):
            d(torch.tensor([0.5, 0.5]))

    def test_dynamic_eps_uses_input_dtype(self) -> None:
        w = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float16)
        d = SumBasedDefuzzifier()
        out = d(w)
        assert not torch.isnan(out).any()
        assert torch.allclose(out.sum(dim=1), torch.ones(2, dtype=torch.float16), atol=1e-3)


class TestLogSumDefuzzifier:
    def test_temperature_1_matches_softmaxlog(self, firing_strengths: torch.Tensor) -> None:
        d1 = LogSumDefuzzifier(temperature=1.0)
        d2 = SoftmaxLogDefuzzifier()
        assert torch.allclose(d1(firing_strengths), d2(firing_strengths), atol=1e-6)

    def test_output_sums_to_one(self, firing_strengths: torch.Tensor) -> None:
        d = LogSumDefuzzifier(temperature=2.0)
        out = d(firing_strengths)
        assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_rejects_non_positive_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature must be positive"):
            LogSumDefuzzifier(temperature=0.0)

    def test_rejects_1d_input(self) -> None:
        d = LogSumDefuzzifier()
        with pytest.raises(ValueError, match="expected w with 2 dims"):
            d(torch.tensor([0.5, 0.5]))


class TestInvLogDefuzzifier:
    def test_output_sums_to_one(self, firing_strengths: torch.Tensor) -> None:
        d = InvLogDefuzzifier()
        out = d(firing_strengths)
        assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_output_shape(self, firing_strengths: torch.Tensor) -> None:
        d = InvLogDefuzzifier()
        assert d(firing_strengths).shape == firing_strengths.shape

    def test_rejects_1d_input(self) -> None:
        d = InvLogDefuzzifier()
        with pytest.raises(ValueError, match="expected w with 2 dims"):
            d(torch.tensor([0.5, 0.5]))

    def test_known_values(self) -> None:
        """w1=e^{-1}, w2=e^{-2} → Z1=-1, Z2=-2 → [2/3, 1/3]."""
        d = InvLogDefuzzifier()
        import math

        w = torch.tensor([[math.e**-1, math.e**-2]])
        out = d(w)
        expected = torch.tensor([[2.0 / 3.0, 1.0 / 3.0]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_scale_invariance(self) -> None:
        """f̄_r is unchanged when all Z_r are multiplied by k>0 (i.e. w → w^k)."""
        d = InvLogDefuzzifier()
        torch.manual_seed(7)
        # Use values safely away from 0 so w**k stays above eps for small k
        w = torch.rand(8, 5) * 0.5 + 0.4  # values in [0.4, 0.9]
        out1 = d(w)
        out2 = d(w**3.0)  # small enough exponent to avoid underflow
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_no_saturation_at_high_dimensions(self) -> None:
        """Max normalized weight stays well below 1 even for D=100, unlike softmax."""
        torch.manual_seed(3)
        N, R, D = 20, 4, 100
        # Design: rule 1 fires at 0.9^D, rules 2-4 fire at 0.5^D
        # Softmax should saturate; InvLog should not
        w_high = torch.full((N, 1), 0.9) ** D
        w_low = torch.full((N, R - 1), 0.5) ** D
        w = torch.cat([w_high, w_low], dim=1)
        d = InvLogDefuzzifier()
        out = d(w)
        assert out.max(dim=1).values.mean().item() < 0.99
