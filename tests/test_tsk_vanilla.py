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
from highfis.models import TSKClassifier, TSKRegressor


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {
        f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)]
        for i in range(n_inputs)
    }


# =====================================================================
# TSKClassifier
# =====================================================================


class TestTSKClassifierInit:
    def test_rejects_empty_input_mfs(self) -> None:
        with pytest.raises(ValueError, match="input_mfs must not be empty"):
            TSKClassifier({}, n_classes=2)

    def test_rejects_invalid_n_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            TSKClassifier(_build_input_mfs(), n_classes=1)

    def test_default_defuzzifier_is_sum_based(self) -> None:
        model = TSKClassifier(_build_input_mfs(), n_classes=3)
        assert isinstance(model.defuzzifier, SumBasedDefuzzifier)


class TestTSKClassifierForward:
    def test_forward_shape(self) -> None:
        model = TSKClassifier(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        logits = model.forward(x)
        assert logits.shape == (8, 3)

    def test_predict_proba_sums_to_one(self) -> None:
        model = TSKClassifier(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        proba = model.predict_proba(x)
        assert proba.shape == (8, 3)
        assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)

    def test_predict_returns_class_indices(self) -> None:
        model = TSKClassifier(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        pred = model.predict(x)
        assert pred.shape == (8,)
        assert pred.dtype == torch.int64

    def test_antecedent_norm_w_sums_to_one(self) -> None:
        """Verify SumBasedDefuzzifier normalizes firing strengths to 1."""
        model = TSKClassifier(_build_input_mfs(), n_classes=2)
        x = torch.randn(6, 3)
        norm_w = model.forward_antecedents(x)
        assert norm_w.ndim == 2
        assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


class TestTSKClassifierFit:
    def test_fit_returns_history(self) -> None:
        torch.manual_seed(1)
        model = TSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,), dtype=torch.long)
        history = model.fit(x, y, epochs=4, learning_rate=1e-2, batch_size=5)
        assert set(history.keys()) == {"train", "ur", "val", "stopped_epoch"}
        assert len(history["train"]) == 4

    def test_fit_with_custom_mse_criterion(self) -> None:
        torch.manual_seed(1)
        model = TSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
        x = torch.randn(16, 2)
        y = torch.randint(0, 2, (16,), dtype=torch.long)
        history = model.fit(x, y, epochs=3, criterion=nn.MSELoss())
        assert len(history["train"]) == 3

    def test_early_stopping_with_val_data(self) -> None:
        torch.manual_seed(42)
        model = TSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
        x = torch.randn(30, 2)
        y = torch.randint(0, 2, (30,), dtype=torch.long)
        x_val = torch.randn(10, 2)
        y_val = torch.randint(0, 2, (10,), dtype=torch.long)
        history = model.fit(
            x, y, epochs=500, x_val=x_val, y_val=y_val, patience=5, learning_rate=1e-2,
        )
        assert len(history["val_acc"]) == len(history["train"])
        assert history["stopped_epoch"] < 500

    def test_fit_validates_x_shape(self) -> None:
        model = TSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
        y = torch.randint(0, 2, (10,), dtype=torch.long)
        with pytest.raises(ValueError, match="expected x shape"):
            model.fit(torch.randn(10, 3), y, epochs=1)

    def test_consequent_batch_norm(self) -> None:
        torch.manual_seed(1)
        model = TSKClassifier(
            _build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2,
            consequent_batch_norm=True,
        )
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,), dtype=torch.long)
        history = model.fit(x, y, epochs=2, batch_size=10)
        assert len(history["train"]) == 2


# =====================================================================
# TSKRegressor
# =====================================================================


class TestTSKRegressorInit:
    def test_rejects_empty_input_mfs(self) -> None:
        with pytest.raises(ValueError, match="input_mfs must not be empty"):
            TSKRegressor({})

    def test_default_defuzzifier_is_sum_based(self) -> None:
        model = TSKRegressor(_build_input_mfs())
        assert isinstance(model.defuzzifier, SumBasedDefuzzifier)


class TestTSKRegressorForward:
    def test_forward_shape(self) -> None:
        model = TSKRegressor(_build_input_mfs())
        x = torch.randn(8, 3)
        out = model.forward(x)
        assert out.shape == (8, 1)

    def test_predict_returns_1d(self) -> None:
        model = TSKRegressor(_build_input_mfs())
        x = torch.randn(8, 3)
        pred = model.predict(x)
        assert pred.shape == (8,)

    def test_antecedent_norm_w_sums_to_one(self) -> None:
        model = TSKRegressor(_build_input_mfs())
        x = torch.randn(6, 3)
        norm_w = model.forward_antecedents(x)
        assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


class TestTSKRegressorFit:
    def test_fit_returns_history(self) -> None:
        torch.manual_seed(1)
        model = TSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
        x = torch.randn(20, 2)
        y = torch.randn(20)
        history = model.fit(x, y, epochs=4, learning_rate=1e-2, batch_size=5)
        assert set(history.keys()) == {"train", "ur", "val", "stopped_epoch"}
        assert len(history["train"]) == 4

    def test_fit_loss_decreases(self) -> None:
        torch.manual_seed(42)
        model = TSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
        x = torch.randn(40, 2)
        y = x[:, 0] + 0.5 * x[:, 1]
        history = model.fit(x, y, epochs=50, learning_rate=1e-2)
        assert history["train"][-1] < history["train"][0]

    def test_early_stopping_with_val_data(self) -> None:
        torch.manual_seed(42)
        model = TSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
        x = torch.randn(30, 2)
        y = x[:, 0] + 0.5 * x[:, 1]
        x_val = torch.randn(10, 2)
        y_val = x_val[:, 0] + 0.5 * x_val[:, 1]
        history = model.fit(
            x, y, epochs=2000, x_val=x_val, y_val=y_val, patience=15, learning_rate=5e-2,
        )
        assert len(history["val"]) > 0
        assert history["stopped_epoch"] < 2000

    def test_constant_targets(self) -> None:
        torch.manual_seed(0)
        model = TSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
        x = torch.randn(20, 2)
        y = torch.full((20,), 3.14)
        model.fit(x, y, epochs=200, learning_rate=5e-2)
        pred = model.predict(x)
        assert float(torch.abs(pred.mean() - 3.14)) < 1.0
