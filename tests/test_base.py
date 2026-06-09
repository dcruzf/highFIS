"""Tests for highfis.base (BaseTSK and helpers)."""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import pytest
import torch
from torch import nn

from highfis.memberships import GaussianMF
from highfis.models._base import BaseTSK
from highfis.optim._utils import (
    _iter_minibatch_indices,
    _log,
    _resolve_verbose,
    _uniform_regularization_loss,
)


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

    def test_resolve_verbose_rejects_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="verbose must be an int in 0\\.\\.3 or a bool"):
            _resolve_verbose(cast(Any, "yes"))

    def test_resolve_verbose_rejects_invalid_range(self) -> None:
        with pytest.raises(ValueError, match="verbose must be between 0 and 3"):
            _resolve_verbose(4)

    def test_log_verbose_logs_message(self) -> None:
        logger = logging.getLogger("test_logger")
        logger.handlers.clear()
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        _log(logger, "verbose test", verbose=True)

    def test_log_verbose_outputs_to_stdout(self) -> None:
        logger = logging.getLogger("test_logger")
        logger.handlers.clear()
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        _log(logger, "verbose test", verbose=True)
        assert logger.handlers
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.handlers[0].stream is sys.stdout

    def test_log_non_verbose_returns_without_logging(self) -> None:
        logger = logging.getLogger("test_logger")
        _log(logger, "quiet test", verbose=False)

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
