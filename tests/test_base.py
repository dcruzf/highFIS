"""Tests for highfis.base (BaseTSK and helpers)."""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

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
        assert torch.allclose(norm_w.sum(dim=1), torch.ones(10), atol=1e-05)

    def test_fit_runs_without_validation(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=5)
        assert "train" in history
        assert "val" not in history
        assert len(history["train"]) == 5

    def test_fit_with_validation(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=5, x_val=x, y_val=y, patience=3)
        assert "val" in history

    def test_fit_with_validation_patience_none(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=5, x_val=x, y_val=y, patience=None)
        assert len(history["train"]) == 5
        assert len(history["val"]) == 5
        assert history["stopped_epoch"] == 5

    def test_fit_with_validation_restore_best_false(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=5, x_val=x, y_val=y, patience=1, restore_best=False)
        assert len(history["train"]) == 5
        assert len(history["val"]) == 5
        assert history["stopped_epoch"] == 5

    def test_fit_verbose_level_one_uses_progress_bar(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=3, verbose=1)
        assert len(history["train"]) == 3

    def test_fit_verbose_level_three_logs_every_epoch(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=3, verbose=3)
        assert len(history["train"]) == 3

    def test_fit_verbose_level_one_with_validation_uses_progress_bar(self, capfd) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=3, verbose=1, x_val=x, y_val=y)
        capfd.readouterr()
        assert len(history["train"]) == 3

    def test_resolve_verbose_rejects_invalid_type(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        with pytest.raises(TypeError, match="verbose must be an int in 0\\.\\.3 or a bool"):
            model._resolve_verbose(cast(Any, "yes"))

    def test_resolve_verbose_rejects_invalid_range(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        with pytest.raises(ValueError, match="verbose must be between 0 and 3"):
            model._resolve_verbose(4)

    def test_log_verbose_logs_message(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        model._log("verbose test", verbose=True)

    def test_log_verbose_outputs_to_stdout(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        model._log("verbose test", verbose=True)
        assert model.logger.handlers
        assert isinstance(model.logger.handlers[0], logging.StreamHandler)
        assert model.logger.handlers[0].stream is sys.stdout

    def test_log_non_verbose_returns_without_logging(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        model._log("quiet test", verbose=False)

    def test_fit_uses_custom_optimizer(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=1, optimizer=optimizer)
        assert "train" in history

    def test_fit_evaluates_custom_metrics(self) -> None:
        model = _ConcreteClassifier(_make_input_mfs(), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        history = model.fit(x, y, epochs=2, metrics=["accuracy", "f1_macro"], x_val=x, y_val=y)
        assert "train_accuracy" in history
        assert "train_f1_macro" in history
        assert "val_accuracy" in history
        assert "val_f1_macro" in history
        assert len(history["train_accuracy"]) == 2
        assert len(history["val_f1_macro"]) == 2

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
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-06)

    def test_uniform_regularization_loss_with_target(self) -> None:
        w = torch.ones(10, 4) / 4.0
        loss = _uniform_regularization_loss(w, target=0.25)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-06)

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


class _ConcreteRegressor(BaseTSK):
    """Minimal concrete regressor subclass for testing."""

    def _build_consequent_layer(self) -> nn.Module:
        from highfis.layers import RegressionConsequentLayer

        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()


def test_get_consequent_weights_none() -> None:
    model = _ConcreteClassifier(_make_input_mfs(), n_classes=3)
    model.consequent_layer = nn.Identity()
    assert model.get_consequent_weights() is None


def test_get_task_by_class_name() -> None:
    class DummyClassifier(BaseTSK):
        def _build_consequent_layer(self) -> nn.Module:
            return nn.Identity()

        def _default_criterion(self) -> nn.Module:
            return nn.MSELoss()

    model = DummyClassifier(_make_input_mfs())
    assert model._get_task() == "classification"


def test_predict_numpy_regression_squeezed() -> None:
    model = _ConcreteRegressor(_make_input_mfs())
    x = torch.randn(10, 2)
    out = model._predict_numpy(x)
    assert out.shape == (10,)


def test_predict_numpy_regression_not_squeezed() -> None:
    from unittest.mock import patch

    model = _ConcreteRegressor(_make_input_mfs())
    x = torch.randn(10, 2)
    with patch.object(model, "forward", return_value=torch.randn(10)):
        out = model._predict_numpy(x)
        assert out.shape == (10,)
    with patch.object(model, "forward", return_value=torch.randn(10, 2)):
        out2 = model._predict_numpy(x)
        assert out2.shape == (10, 2)
