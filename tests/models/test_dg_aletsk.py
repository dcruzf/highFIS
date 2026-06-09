from __future__ import annotations

import pytest
import torch

from highfis.layers import (
    DGALETSKRuleLayer,
    GatedClassificationConsequentLayer,
    GatedClassificationZeroOrderConsequentLayer,
    GatedRegressionConsequentLayer,
    GatedRegressionZeroOrderConsequentLayer,
)
from highfis.memberships import GaussianMF
from highfis.models import DGALETSKClassifierModel, DGALETSKRegressorModel


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_dgaletsk_classifier_forward_shapes() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(5, 3)
    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)
    assert logits.shape == (5, 3)
    assert proba.shape == (5, 3)
    assert pred.shape == (5,)


def test_dgaletsk_regressor_forward_shape() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    output = model.forward(x)
    assert output.shape == (4, 1)


def test_dgaletsk_classifier_architecture() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=3)
    assert isinstance(model.rule_layer, DGALETSKRuleLayer)
    assert isinstance(model.consequent_layer, GatedClassificationZeroOrderConsequentLayer)
    theta_before = model.consequent_layer.theta_gates.detach().clone()
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_dgaletsk_regressor_architecture() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    assert isinstance(model.rule_layer, DGALETSKRuleLayer)
    assert isinstance(model.consequent_layer, GatedRegressionZeroOrderConsequentLayer)
    theta_before = model.consequent_layer.theta_gates.detach().clone()
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_dgaletsk_thresholds_and_pruning() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.rule_layer.lambda_gates.data.fill_(0.0)
    model.rule_layer.lambda_gates.data[0] = 1.0
    model.consequent_layer.theta_gates.data.fill_(0.0)
    model.consequent_layer.theta_gates.data[0] = 1.0
    tau_lambda, tau_theta = model.compute_thresholds(0.5, 0.5)
    fv = model.get_feature_gate_values().detach()
    rv = model.get_rule_gate_values().detach()
    expected_tau_lambda = float(fv.max()) - 0.5 * float(fv.max() - fv.min())
    expected_tau_theta = float(rv.max()) - 0.5 * float(rv.max() - rv.min())
    assert abs(tau_lambda - expected_tau_lambda) < 1e-05
    assert abs(tau_theta - expected_tau_theta) < 1e-05
    model.apply_thresholds(tau_lambda, tau_theta)
    assert model.rule_layer.lambda_gates.data[0] == 1.0
    assert model.consequent_layer.theta_gates.data[0] == 1.0


def test_dgaletsk_classifier_invalid_zeta_raises() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="zeta must be in \\[0, 1\\]"):
        model.compute_thresholds(-0.1, 0.5)


def test_dgaletsk_classifier_apply_thresholds_invalid_thresholds_raises() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="thresholds must be finite"):
        model.apply_thresholds(float("nan"), 0.0)


def test_dgaletsk_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DGALETSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_dgaletsk_rule_layer_lambda_gates_shape_is_per_feature() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=4, n_mfs=2), n_classes=2)
    assert model.rule_layer.lambda_gates.shape == (4,)


def test_dgaletsk_regressor_lambda_gates_shape_is_per_feature() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=5, n_mfs=2))
    assert model.rule_layer.lambda_gates.shape == (5,)


def test_dgaletsk_classifier_first_order_consequent_mode_is_re() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    assert model.consequent_layer.mode == "re"


def test_dgaletsk_regressor_first_order_consequent_mode_is_re() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    assert model.consequent_layer.mode == "re"


def test_dgaletsk_rule_layer_firing_strengths_in_unit_interval() -> None:
    layer = DGALETSKRuleLayer(input_names=["x1", "x2", "x3"], mf_per_input=[2, 2, 2], rule_base="coco")
    layer.lambda_gates.data.fill_(1.0)
    mf_outputs = {"x1": torch.tensor([[0.8, 0.9]]), "x2": torch.tensor([[0.5, 0.7]]), "x3": torch.tensor([[0.3, 0.6]])}
    with torch.no_grad():
        f = layer(mf_outputs)
    assert f.shape == (1, 2)
    assert torch.allclose(f[0, 0], torch.tensor(0.3), atol=0.01)
    assert torch.allclose(f[0, 1], torch.tensor(0.6), atol=0.01)


def test_dgaletsk_classifier_prune_structure_empty_features_raises() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="surviving_features must not be empty"):
        model.prune_structure([], [0])


def test_dgaletsk_classifier_prune_structure_empty_rules_raises() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="surviving_rules must not be empty"):
        model.prune_structure([0], [])


def test_dgaletsk_regressor_prune_structure_empty_features_raises() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    with pytest.raises(ValueError, match="surviving_features must not be empty"):
        model.prune_structure([], [0])


def test_dgaletsk_regressor_prune_structure_empty_rules_raises() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    with pytest.raises(ValueError, match="surviving_rules must not be empty"):
        model.prune_structure([0], [])


def test_dgaletsk_regressor_search_thresholds_sf_non_empty_no_fallback() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.rule_layer.lambda_gates.data = torch.tensor([1.0, 0.5])
    model.consequent_layer.theta_gates.data = torch.tensor([1.0, 0.5])
    x = torch.randn(16, 2)
    y = torch.randn(16)
    result = model.search_thresholds(
        x, y, zeta_lambda=[1.0], zeta_theta=[1.0], inplace=True, structural=True, use_lse=False
    )
    assert result["surviving_feature_indices"] == [0]
    assert result["surviving_rule_indices"] == [0]


def test_dg_aletsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=1)


def test_dg_aletsk_classifier_thresholds_and_convert() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    assert isinstance(model.consequent_layer, GatedClassificationZeroOrderConsequentLayer)
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)
    tau_lambda, tau_theta = model.compute_thresholds(0.5, 0.5)
    assert isinstance(tau_lambda, float)
    assert isinstance(tau_theta, float)
    assert tau_lambda == tau_lambda
    assert tau_theta == tau_theta
    with pytest.raises(ValueError, match="thresholds must be finite"):
        model.apply_thresholds(float("nan"), 0.0)


def test_dg_aletsk_classifier_search_thresholds_inplace_false() -> None:
    torch.manual_seed(0)
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=False, verbose=True)
    after = model.state_dict()
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_dg_aletsk_classifier_search_thresholds_inplace_true_converts_zero_order_self() -> None:
    torch.manual_seed(0)
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)
    assert isinstance(model.consequent_layer, GatedClassificationZeroOrderConsequentLayer)
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=True, verbose=False)
    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_classifier_convert_to_first_order_idempotent() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    model.convert_to_first_order()
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)


def test_dg_aletsk_regressor_thresholds_apply_and_search() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]
    with pytest.raises(ValueError, match="thresholds must be finite"):
        model.apply_thresholds(float("inf"), 0.0)
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=False)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_regressor_search_thresholds_inplace_true_and_verbose() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=False, inplace=True, verbose=True)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_regressor_search_thresholds_inplace_true_keeps_zero_order_when_no_lse() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]
    assert isinstance(model.consequent_layer, GatedRegressionZeroOrderConsequentLayer)
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=False, inplace=True, verbose=False)
    assert isinstance(model.consequent_layer, GatedRegressionZeroOrderConsequentLayer)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_regressor_convert_to_first_order_idempotent() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)


def test_dg_aletsk_classifier_convert_to_first_order_preserves_theta() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    assert isinstance(model.consequent_layer, GatedClassificationZeroOrderConsequentLayer)
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)


def test_dg_aletsk_regressor_convert_to_first_order_preserves_theta() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    assert isinstance(model.consequent_layer, GatedRegressionZeroOrderConsequentLayer)
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)


def test_dg_aletsk_regressor_search_thresholds_verbose() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=False, verbose=True)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_regressor_search_thresholds_use_lse_false() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=False, inplace=False, verbose=False)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_classifier_search_thresholds_no_candidates_raises() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,), dtype=torch.long)
    with pytest.raises(RuntimeError, match="threshold search did not yield a valid candidate"):
        model.search_thresholds(x, y, zeta_lambda=[], zeta_theta=[], inplace=False)


def test_dg_aletsk_regressor_search_thresholds_no_candidates_raises() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(8, 2)
    y = torch.randn(8)
    with pytest.raises(RuntimeError, match="threshold search did not yield a valid candidate"):
        model.search_thresholds(x, y, zeta_lambda=[], zeta_theta=[], inplace=False)


def test_dg_aletsk_classifier_convert_to_first_order_preserves_theta_values() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    theta_before = model.consequent_layer.theta_gates.detach().clone()
    model.convert_to_first_order()
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_dg_aletsk_regressor_convert_to_first_order_preserves_theta_values() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    theta_before = model.consequent_layer.theta_gates.detach().clone()
    model.convert_to_first_order()
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_dg_aletsk_classifier_search_thresholds_inplace_true_loads_state() -> None:
    torch.manual_seed(0)
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    model.convert_to_first_order()
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=True, verbose=False)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}
    assert isinstance(result["best_score"], float)


def test_dg_aletsk_regressor_convert_to_first_order_and_search_inplace_true() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=True)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgaletsk_init_consequents_raises_after_conversion() -> None:
    mfs = {"x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]}
    model_dga = DGALETSKClassifierModel(mfs, n_classes=2)
    model_dga.convert_to_first_order()
    with pytest.raises(ValueError, match="requires a zero-order consequent layer"):
        model_dga.init_consequents_from_labels(torch.tensor([0, 1]))
