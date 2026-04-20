from __future__ import annotations

import torch

from highfis.layers import (
    DGALETSKRuleLayer,
    GatedClassificationConsequentLayer,
    GatedClassificationZeroOrderConsequentLayer,
    GatedRegressionConsequentLayer,
    GatedRegressionZeroOrderConsequentLayer,
)
from highfis.memberships import GaussianMF
from highfis.models import DGALETSKClassifier, DGALETSKRegressor


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_dgaletsk_classifier_forward_shapes() -> None:
    model = DGALETSKClassifier(_build_input_mfs(), n_classes=3)
    x = torch.randn(5, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (5, 3)
    assert proba.shape == (5, 3)
    assert pred.shape == (5,)


def test_dgaletsk_regressor_forward_shape() -> None:
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)

    output = model.forward(x)

    assert output.shape == (4, 1)


def test_dgaletsk_classifier_architecture() -> None:
    model = DGALETSKClassifier(_build_input_mfs(), n_classes=3)

    assert isinstance(model.rule_layer, DGALETSKRuleLayer)
    assert isinstance(model.consequent_layer, GatedClassificationZeroOrderConsequentLayer)
    assert model.rule_layer.alpha.item() > 0.0

    theta_before = model.consequent_layer.theta_gates.detach().clone()
    model.convert_to_first_order()

    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_dgaletsk_regressor_architecture() -> None:
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))

    assert isinstance(model.rule_layer, DGALETSKRuleLayer)
    assert isinstance(model.consequent_layer, GatedRegressionZeroOrderConsequentLayer)

    theta_before = model.consequent_layer.theta_gates.detach().clone()
    model.convert_to_first_order()

    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)
