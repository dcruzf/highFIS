"""Tests for highfis.base (BaseTSK and helpers)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from highfis.base import BaseTSK, _iter_minibatch_indices, _uniform_regularization_loss
from highfis.memberships import GaussianMF


class _ConcreteClassifier(BaseTSK):
    """Minimal concrete subclass for testing."""

    def __init__(self, input_mfs, n_classes: int = 3, **kwargs):
        self._n_classes = n_classes
        super().__init__(input_mfs, **kwargs)

    def _build_consequent_layer(self) -> nn.Module:
        from highfis.layers import ClassificationConsequentLayer

        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self._n_classes)

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()


def _make_input_mfs():
    return {"x1": [GaussianMF(0.0, 1.0), GaussianMF(1.0, 1.0)], "x2": [GaussianMF(0.0, 1.0), GaussianMF(1.0, 1.0)]}


class TestBaseTSK:
    def test_forward_shape(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=3)
        x = torch.randn(10, 2)
        out = model(x)
        assert out.shape == (10, 3)

    def test_forward_antecedents_shape(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=3)
        x = torch.randn(10, 2)
        norm_w = model.forward_antecedents(x)
        assert norm_w.shape == (10, model.n_rules)
        assert torch.allclose(norm_w.sum(dim=1), torch.ones(10), atol=1e-5)

    def test_fit_runs_without_validation(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=5)
        assert "train" in history
        assert len(history["train"]) == 5

    def test_fit_with_validation(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=5, x_val=x, y_val=y, patience=3)
        assert "val" in history

    def test_rejects_empty_input_mfs(self) -> None:
        with pytest.raises(ValueError, match="input_mfs must not be empty"):
            _ConcreteClassifier({}, n_classes=2)

    def test_custom_defuzzifier(self) -> None:
        from highfis.defuzzifiers import SumBasedDefuzzifier

        model = _ConcreteClassifier(_make_input_mfs(), n_classes=3, defuzzifier=SumBasedDefuzzifier())
        x = torch.randn(5, 2)
        out = model(x)
        assert out.shape == (5, 3)


class TestHelpers:
    def test_uniform_regularization_loss(self) -> None:
        w = torch.ones(10, 4) / 4.0
        loss = _uniform_regularization_loss(w)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)

    def test_uniform_regularization_loss_with_target(self) -> None:
        w = torch.ones(10, 4) / 4.0
        loss = _uniform_regularization_loss(w, target=0.25)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)

    def test_iter_minibatch_indices_no_batch(self) -> None:
        batches = _iter_minibatch_indices(100, None, False)
        assert len(batches) == 1
        assert len(batches[0]) == 100

    def test_iter_minibatch_indices_with_batch(self) -> None:
        batches = _iter_minibatch_indices(100, 30, False)
        assert len(batches) == 4
        total = sum(len(b) for b in batches)
        assert total == 100

    def test_iter_minibatch_rejects_zero_batch(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            _iter_minibatch_indices(10, 0, False)
