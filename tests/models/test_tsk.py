"""Tests for vanilla TSK models (Takagi & Sugeno, 1985).

These models use the product t-norm and SumBasedDefuzzifier:

    w_r = ∏_{d=1}^{D} μ_{r,d}(x_d)
    f̄_r = w_r / Σ w_i
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from highfis.defuzzifiers import SumBasedDefuzzifier
from highfis.memberships import GaussianMF
from highfis.models import TSKClassifierModel, TSKRegressorModel


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


class TestTSKClassifierModelInit:
    def test_rejects_empty_input_mfs(self) -> None:
        with pytest.raises(ValueError, match="input_mfs must not be empty"):
            TSKClassifierModel({}, n_classes=2)

    def test_rejects_invalid_n_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            TSKClassifierModel(_build_input_mfs(), n_classes=1)

    def test_default_defuzzifier_is_sum_based(self) -> None:
        model = TSKClassifierModel(_build_input_mfs(), n_classes=3)
        assert isinstance(model.defuzzifier, SumBasedDefuzzifier)


class TestTSKClassifierModelForward:
    def test_forward_shape(self) -> None:
        model = TSKClassifierModel(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        logits = model.forward(x)
        assert logits.shape == (8, 3)

    def test_predict_proba_sums_to_one(self) -> None:
        model = TSKClassifierModel(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        proba = model.predict_proba(x)
        assert proba.shape == (8, 3)
        assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-06)

    def test_predict_returns_class_indices(self) -> None:
        model = TSKClassifierModel(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        pred = model.predict(x)
        assert pred.shape == (8,)
        assert pred.dtype == torch.int64

    def test_antecedent_norm_w_sums_to_one(self) -> None:
        """Verify SumBasedDefuzzifier normalizes firing strengths to 1."""
        model = TSKClassifierModel(_build_input_mfs(), n_classes=2)
        x = torch.randn(6, 3)
        norm_w = model.forward_antecedents(x)
        assert norm_w.ndim == 2
        assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-06)


class TestTSKRegressorInit:
    def test_rejects_empty_input_mfs(self) -> None:
        with pytest.raises(ValueError, match="input_mfs must not be empty"):
            TSKRegressorModel({})

    def test_default_defuzzifier_is_sum_based(self) -> None:
        model = TSKRegressorModel(_build_input_mfs())
        assert isinstance(model.defuzzifier, SumBasedDefuzzifier)


class TestTSKRegressorForward:
    def test_forward_shape(self) -> None:
        model = TSKRegressorModel(_build_input_mfs())
        x = torch.randn(8, 3)
        out = model.forward(x)
        assert out.shape == (8, 1)

    def test_predict_returns_1d(self) -> None:
        model = TSKRegressorModel(_build_input_mfs())
        x = torch.randn(8, 3)
        pred = model.predict(x)
        assert pred.shape == (8,)

    def test_antecedent_norm_w_sums_to_one(self) -> None:
        model = TSKRegressorModel(_build_input_mfs())
        x = torch.randn(6, 3)
        norm_w = model.forward_antecedents(x)
        assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-06)


def test_tsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        TSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_tsk_regressor_forward_shape() -> None:
    model = TSKRegressorModel(_build_input_mfs())
    x = torch.randn(4, 3)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_tsk_classifier_forward_shapes() -> None:
    model = TSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(4, 3)
    logits = model.forward(x)
    assert logits.shape == (4, 2)
    assert torch.allclose(torch.softmax(logits, dim=1).sum(dim=1), torch.ones(4), atol=1e-06)


def test_tsk_classifier_default_criterion() -> None:
    model = TSKClassifierModel(_build_input_mfs(), n_classes=2)
    assert isinstance(model._default_criterion(), nn.CrossEntropyLoss)


def test_tsk_regressor_default_criterion() -> None:
    model = TSKRegressorModel(_build_input_mfs())
    assert isinstance(model._default_criterion(), nn.MSELoss)
