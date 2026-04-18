"""Tests for highfis.defuzzifiers."""

from __future__ import annotations

import pytest
import torch

from highfis.defuzzifiers import LogSumDefuzzifier, SoftmaxLogDefuzzifier, SumBasedDefuzzifier


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
