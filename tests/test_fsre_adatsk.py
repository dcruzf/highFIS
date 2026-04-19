from __future__ import annotations

import torch

from highfis.layers import AdaSoftminRuleLayer
from highfis.memberships import GaussianMF
from highfis.models import FSREAdaTSKClassifier, FSREAdaTSKRegressor


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_fsre_adatsk_classifier_forward_shapes() -> None:
    model = FSREAdaTSKClassifier(_build_input_mfs(), n_classes=3)
    x = torch.randn(5, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (5, 3)
    assert proba.shape == (5, 3)
    assert pred.shape == (5,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(5), atol=1e-6)


def test_fsre_adatsk_regressor_forward_shape() -> None:
    model = FSREAdaTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)

    output = model.forward(x)

    assert output.shape == (4, 1)


def test_fsre_adatsk_forward_antecedents_row_sum_one() -> None:
    model = FSREAdaTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(6, 2)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_fsre_adatsk_expand_to_en_frb_increases_rule_count() -> None:
    model = FSREAdaTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    initial_rules = model.n_rules

    model.expand_to_en_frb()

    assert model.n_rules > initial_rules
    assert isinstance(model.rule_layer, AdaSoftminRuleLayer)
