from __future__ import annotations

import pytest
import torch

from highfis.memberships import ConstantMF, GaussianMF
from highfis.models import MHTSKClassifierModel, MHTSKRegressorModel, build_rule_feature_mask


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_mhtsk_classifier_sparse_consequent_forward_shape() -> None:
    input_mfs = {
        "x1": [GaussianMF(mean=-1.0, sigma=0.5), GaussianMF(mean=1.0, sigma=0.5), ConstantMF(1.0)],
        "x2": [GaussianMF(mean=-1.0, sigma=0.5), GaussianMF(mean=1.0, sigma=0.5), ConstantMF(1.0)],
    }
    rules = [(0, 2), (2, 1)]
    rule_feature_mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    model = MHTSKClassifierModel(input_mfs, rule_feature_mask, rules, n_classes=2)
    x = torch.randn(4, 2)
    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)
    assert logits.shape == (4, 2)
    assert proba.shape == (4, 2)
    assert pred.shape == (4,)


def test_mhtsk_regressor_sparse_consequent_forward_shape() -> None:
    input_mfs = {
        "x1": [GaussianMF(mean=-1.0, sigma=0.5), GaussianMF(mean=1.0, sigma=0.5), ConstantMF(1.0)],
        "x2": [GaussianMF(mean=-1.0, sigma=0.5), GaussianMF(mean=1.0, sigma=0.5), ConstantMF(1.0)],
    }
    rules = [(0, 2), (2, 1)]
    rule_feature_mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    model = MHTSKRegressorModel(input_mfs, rule_feature_mask, rules)
    x = torch.randn(4, 2)
    output = model.forward(x)
    pred = model.predict(x)
    assert output.shape == (4, 1)
    assert pred.shape == (4,)


def test_build_rule_feature_mask_valid() -> None:
    rules = [(0, 1), (1, 0)]
    dont_care_indices = [0, 0]
    mask = build_rule_feature_mask(rules, dont_care_indices)
    assert mask.shape == (2, 2)
    assert mask.tolist() == [[False, True], [True, False]]


def test_build_rule_feature_mask_rejects_invalid_rules() -> None:
    with pytest.raises(ValueError, match="rules must not be empty"):
        build_rule_feature_mask([], [0, 0])
    with pytest.raises(ValueError, match="dont_care_indices must match the rule input dimension"):
        build_rule_feature_mask([(0, 1)], [0])
    with pytest.raises(ValueError, match="all rules must have the same length"):
        build_rule_feature_mask([(0, 1), (0, 1, 2)], [0, 0])


def test_mhtsk_classifier_rejects_invalid_n_classes() -> None:
    input_mfs = {
        "x1": [GaussianMF(mean=-1.0, sigma=0.5), ConstantMF(1.0)],
        "x2": [GaussianMF(mean=-1.0, sigma=0.5), ConstantMF(1.0)],
    }
    rules = [(0, 1), (1, 0)]
    rule_feature_mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        MHTSKClassifierModel(input_mfs, rule_feature_mask, rules, n_classes=1)


def test_mhtsk_default_criteria() -> None:
    from torch import nn

    input_mfs = {
        "x1": [GaussianMF(mean=-1.0, sigma=0.5), ConstantMF(1.0)],
        "x2": [GaussianMF(mean=-1.0, sigma=0.5), ConstantMF(1.0)],
    }
    rules = [(0, 1), (1, 0)]
    rule_feature_mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    clf = MHTSKClassifierModel(input_mfs, rule_feature_mask, rules, n_classes=2)
    reg = MHTSKRegressorModel(input_mfs, rule_feature_mask, rules)

    # Paper-faithful: MHTSK uses MSE on one-hot targets (Bian et al. 2025, eq. 12).
    assert isinstance(clf.default_criterion(), nn.MSELoss)
    assert isinstance(reg.default_criterion(), nn.MSELoss)
