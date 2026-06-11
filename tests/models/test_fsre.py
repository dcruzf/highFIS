from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from highfis.layers import AdaSoftminRuleLayer
from highfis.memberships import GaussianMF
from highfis.models import FSREADATSKClassifierModel, FSREADATSKRegressorModel


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_fsre_adatsk_classifier_forward_shapes() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(5, 3)
    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)
    assert logits.shape == (5, 3)
    assert proba.shape == (5, 3)
    assert pred.shape == (5,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(5), atol=1e-06)


def test_fsre_adatsk_regressor_forward_shape() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    output = model.forward(x)
    assert output.shape == (4, 1)


def test_fsre_adatsk_forward_antecedents_row_sum_one() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(6, 2)
    norm_w = model.forward_antecedents(x)
    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-06)


def test_fsre_adatsk_expand_to_en_frb_increases_rule_count() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    initial_rules = model.n_rules
    model.expand_to_en_frb()
    assert model.n_rules > initial_rules
    assert isinstance(model.rule_layer, AdaSoftminRuleLayer)


def test_fsre_adatsk_classifier_lambda_gates_shared_per_feature() -> None:
    n_inputs = 4
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=n_inputs, n_mfs=2), n_classes=2)
    lam = model.consequent_layer.lambda_gates
    assert lam.shape == (n_inputs,)


def test_fsre_adatsk_regressor_lambda_gates_shared_per_feature() -> None:
    n_inputs = 3
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=n_inputs, n_mfs=2))
    lam = model.consequent_layer.lambda_gates
    assert lam.shape == (n_inputs,)


def test_fsre_adatsk_classifier_initial_mode_is_fs() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(), n_classes=2)
    assert model.consequent_layer.mode == "fs"


def test_fsre_adatsk_regressor_initial_mode_is_fs() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs())
    assert model.consequent_layer.mode == "fs"


def test_fsre_adatsk_classifier_expand_to_en_frb_resets_mode_to_fs() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    model.expand_to_en_frb()
    assert model.consequent_layer.mode == "fs"


def test_fsre_adatsk_classifier_mode_fs_uses_only_feature_gates() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "fs"
    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)
    assert torch.allclose(out_before, out_after)


def test_fsre_adatsk_classifier_mode_re_uses_only_rule_gates() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "re"
    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        out_after = model.forward(x)
    assert torch.allclose(out_before, out_after)


def test_fsre_adatsk_classifier_mode_finetune_ignores_all_gates() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "finetune"
    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)
    assert torch.allclose(out_before, out_after)


def test_get_feature_gate_values_shape_classifier() -> None:
    n_inputs = 4
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=n_inputs, n_mfs=2), n_classes=2)
    vals = model.get_feature_gate_values()
    assert vals.shape == (n_inputs,)
    assert not vals.requires_grad


def test_get_rule_gate_values_shape_regressor() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=3, n_mfs=2))
    model.expand_to_en_frb()
    vals = model.get_rule_gate_values()
    assert vals.shape == (model.n_rules,)
    assert not vals.requires_grad


def test_prune_to_features_updates_model_attributes_classifier() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=4, n_mfs=2), n_classes=2)
    model.prune_to_features([0, 2])
    assert model.n_inputs == 2
    assert model.input_names == ["x1", "x3"]
    assert list(model.input_mfs.keys()) == ["x1", "x3"]


def test_prune_to_features_updates_model_attributes_regressor() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=4, n_mfs=2))
    model.prune_to_features([1, 3])
    assert model.n_inputs == 2
    assert model.input_names == ["x2", "x4"]


def test_prune_to_features_empty_raises_classifier() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="must not be empty"):
        model.prune_to_features([])


def test_prune_to_features_then_expand_en_frb_uses_surviving_features() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=4, n_mfs=2), n_classes=2)
    model.prune_to_features([0, 1])
    assert model.n_inputs == 2
    model.expand_to_en_frb()
    assert model.n_rules == 2 * 2
    x = torch.randn(5, 2)
    assert model.forward(x).shape == (5, 2)


def test_prune_to_rules_classifier_updates_n_rules() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=3, n_mfs=2), n_classes=2)
    model.expand_to_en_frb()
    n_before = model.n_rules
    model.prune_to_rules([0, 1])
    assert model.n_rules == 2
    assert model.n_rules < n_before
    assert model.consequent_layer.mode == "finetune"


def test_prune_to_rules_regressor_copies_bias() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.expand_to_en_frb()
    old_bias = model.consequent_layer.bias.data.clone()
    surviving = [0, 2]
    model.prune_to_rules(surviving)
    assert torch.allclose(model.consequent_layer.bias.data, old_bias[surviving])


def test_prune_to_rules_empty_raises_regressor() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs())
    model.expand_to_en_frb()
    with pytest.raises(ValueError, match="must not be empty"):
        model.prune_to_rules([])


def test_prune_to_rules_empty_raises_classifier() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=3, n_mfs=2), n_classes=2)
    model.expand_to_en_frb()
    with pytest.raises(ValueError, match="must not be empty"):
        model.prune_to_rules([])


def test_fsre_adatsk_regressor_mode_fs_uses_only_feature_gates() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "fs"
    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)
    assert torch.allclose(out_before, out_after)


def test_fsre_adatsk_regressor_mode_re_uses_only_rule_gates() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "re"
    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        out_after = model.forward(x)
    assert torch.allclose(out_before, out_after)


def test_fsre_adatsk_regressor_mode_finetune_ignores_all_gates() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "finetune"
    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)
    assert torch.allclose(out_before, out_after)


def test_prune_to_features_with_consequent_bn_classifier() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=3, n_mfs=2), n_classes=2, consequent_batch_norm=True)
    assert model.consequent_batch_norm is True
    model.prune_to_features([0, 2])
    assert model.n_inputs == 2
    assert isinstance(model.consequent_bn, nn.BatchNorm1d)
    assert model.consequent_bn.num_features == 2


def test_prune_to_features_with_consequent_bn_regressor() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=3, n_mfs=2), consequent_batch_norm=True)
    assert model.consequent_batch_norm is True
    model.prune_to_features([1])
    assert model.n_inputs == 1
    assert isinstance(model.consequent_bn, nn.BatchNorm1d)
    assert model.consequent_bn.num_features == 1


def test_prune_to_features_empty_raises_regressor() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    with pytest.raises(ValueError, match="surviving_features must not be empty"):
        model.prune_to_features([])


def test_fsre_adatsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=1)


def test_fsre_adatsk_default_criterion() -> None:
    model_clf = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    assert isinstance(model_clf._default_criterion(), nn.CrossEntropyLoss)

    model_reg = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    assert isinstance(model_reg._default_criterion(), nn.MSELoss)
