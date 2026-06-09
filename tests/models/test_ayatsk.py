from __future__ import annotations

import pytest
import torch
from torch import nn

from highfis.memberships import GaussianMF
from highfis.models import AYATSKClassifierModel, AYATSKRegressorModel
from highfis.models._yager import _adaptive_yager_lambda, _zero_initialize_consequents


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_ayatsk_classifier_forward_predict_shapes() -> None:
    model = AYATSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)
    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)
    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-06)


def test_ayatsk_regressor_forward_predict_shape() -> None:
    model = AYATSKRegressorModel(_build_input_mfs(), rule_base="coco")
    x = torch.randn(6, 3)
    output = model.forward(x)
    pred = model.predict(x)
    assert output.shape == (6, 1)
    assert pred.shape == (6,)


def test_ayatsk_classifier_init_validates_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        AYATSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_ayatsk_classifier_zero_initializes_consequents() -> None:
    model = AYATSKClassifierModel(_build_input_mfs(), n_classes=3)
    weight = getattr(model.consequent_layer, "weight", None)
    bias = getattr(model.consequent_layer, "bias", None)
    assert isinstance(weight, torch.Tensor)
    assert isinstance(bias, torch.Tensor)
    assert torch.allclose(weight, torch.zeros_like(weight))
    assert torch.allclose(bias, torch.zeros_like(bias))


def test_ayatsk_classifier_uses_adaptive_yager_lambda() -> None:
    model = AYATSKClassifierModel(_build_input_mfs(), n_classes=3)
    assert model.lambda_ > 0.0
    assert model.lower_bound_ > 0.0


def test_ayatsk_regressor_zero_initializes_consequents() -> None:
    model = AYATSKRegressorModel(_build_input_mfs(), rule_base="coco")
    weight = getattr(model.consequent_layer, "weight", None)
    bias = getattr(model.consequent_layer, "bias", None)
    assert isinstance(weight, torch.Tensor)
    assert isinstance(bias, torch.Tensor)
    assert torch.allclose(weight, torch.zeros_like(weight))
    assert torch.allclose(bias, torch.zeros_like(bias))


def test_adaptive_yager_lambda_validates_inputs() -> None:
    with pytest.raises(ValueError, match="dimension must be > 1"):
        _adaptive_yager_lambda(1, 0.2)
    with pytest.raises(ValueError, match="lower_bound must be in \\(0, 1\\)"):
        _adaptive_yager_lambda(3, 1.0)


def test_yager_zero_initialize_consequents_handles_missing_params() -> None:
    class DummyLayer(nn.Module):
        pass

    _zero_initialize_consequents(DummyLayer())
