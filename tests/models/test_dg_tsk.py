from __future__ import annotations

import pytest
import torch

from highfis.memberships import GaussianMF
from highfis.models import DGTSKClassifierModel, DGTSKRegressorModel


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_dgtsk_classifier_forward_shapes() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(5, 3)
    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)
    assert logits.shape == (5, 3)
    assert proba.shape == (5, 3)
    assert pred.shape == (5,)


def test_dgtsk_regressor_forward_shape() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    output = model.forward(x)
    assert output.shape == (4, 1)


def test_dgtsk_classifier_architecture() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=3)
    assert model.rule_layer.__class__.__name__ == "DGTSKRuleLayer"
    assert model.consequent_layer.__class__.__name__ == "GatedClassificationZeroOrderConsequentLayer"


def test_dgtsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DGTSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_dgtsk_classifier_apply_thresholds_invalid_thresholds_raises() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="thresholds must be finite"):
        model.apply_thresholds(float("nan"), 0.0)


def test_dgtsk_classifier_convert_to_first_order_preserves_theta() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    theta_before = model.consequent_layer.theta_gates.detach().clone()
    model.convert_to_first_order()
    assert model.consequent_layer.__class__.__name__ == "GatedClassificationConsequentLayer"
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_dgtsk_regressor_convert_to_first_order_preserves_theta() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    theta_before = model.consequent_layer.theta_gates.detach().clone()
    model.convert_to_first_order()
    assert model.consequent_layer.__class__.__name__ == "GatedRegressionConsequentLayer"
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_dgtsk_classifier_thresholds_and_pruning() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.rule_layer.lambda_gates.data.fill_(0.0)
    model.rule_layer.lambda_gates.data[0] = 1.0
    model.consequent_layer.theta_gates.data.fill_(0.0)
    model.consequent_layer.theta_gates.data[0] = 1.0
    tau_lambda, tau_theta = model.compute_thresholds(0.5, 0.5)
    assert torch.isclose(torch.tensor(tau_lambda), torch.tensor(0.5))
    assert torch.isclose(torch.tensor(tau_theta), torch.tensor(0.5))
    model.apply_thresholds(tau_lambda, tau_theta)
    assert model.rule_layer.lambda_gates.data[0] == 1.0
    assert model.consequent_layer.theta_gates.data[0] == 1.0


def test_dgtsk_classifier_search_thresholds_on_first_order_model() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    result = model.search_thresholds(x, y, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], inplace=True, verbose=False)
    assert model.consequent_layer.__class__.__name__ == "GatedClassificationConsequentLayer"
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgtsk_regressor_search_thresholds_on_first_order_model() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    x = torch.randn(20, 2)
    y = torch.randn(20)
    result = model.search_thresholds(x, y, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], inplace=True, verbose=False)
    assert model.consequent_layer.__class__.__name__ == "GatedRegressionConsequentLayer"
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgtsk_classifier_convert_to_first_order_idempotent() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    model.convert_to_first_order()
    assert model.consequent_layer.__class__.__name__ == "GatedClassificationConsequentLayer"


def test_dgtsk_regressor_convert_to_first_order_idempotent() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    model.convert_to_first_order()
    assert model.consequent_layer.__class__.__name__ == "GatedRegressionConsequentLayer"


def test_dgtsk_regressor_search_thresholds_no_candidates_raises() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(8, 2)
    y = torch.randn(8)
    with pytest.raises(RuntimeError, match="threshold search did not yield a valid candidate"):
        model.search_thresholds(x, y, zeta_lambda=[], zeta_theta=[], inplace=False)


def test_dgtsk_classifier_search_thresholds_no_candidates_raises() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,))
    with pytest.raises(RuntimeError, match="threshold search did not yield a valid candidate"):
        model.search_thresholds(x, y, zeta_lambda=[], zeta_theta=[], inplace=False)


def test_dgtsk_regressor_apply_thresholds_invalid_thresholds_raises() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    with pytest.raises(ValueError, match="thresholds must be finite"):
        model.apply_thresholds(float("nan"), 0.0)


def test_dgtsk_classifier_lambda_gates_shape_is_per_feature() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(n_inputs=4, n_mfs=2), n_classes=2)
    assert model.rule_layer.lambda_gates.shape == (4,)


def test_dgtsk_regressor_lambda_gates_shape_is_per_feature() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=5, n_mfs=2))
    assert model.rule_layer.lambda_gates.shape == (5,)


def test_dgtsk_classifier_first_order_consequent_mode_is_re() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    assert model.consequent_layer.mode == "re"


def test_dgtsk_regressor_first_order_consequent_mode_is_re() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    assert model.consequent_layer.mode == "re"


def test_dgtsk_classifier_init_consequents_from_labels() -> None:
    mfs = {f"x{i}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(2)] for i in range(2)}
    model = DGTSKClassifierModel(mfs, n_classes=4, rule_base="coco")
    n_rules = model.n_rules
    y = torch.arange(min(n_rules, 4), dtype=torch.long)
    model.init_consequents_from_labels(y)
    expected = torch.zeros(n_rules, 4)
    n = min(len(y), n_rules)
    expected[:n].scatter_(1, y[:n].unsqueeze(1), 1.0)
    assert torch.allclose(model.consequent_layer.bias.data, expected)


def test_dgtsk_classifier_init_consequents_from_labels_partial_fill() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=3), n_classes=2)
    model.init_consequents_from_labels(torch.tensor([1], dtype=torch.long))
    assert model.consequent_layer.bias.data[0, 1].item() == 1.0
    assert model.consequent_layer.bias.data[1:].abs().sum().item() == 0.0


def test_dgtsk_classifier_init_consequents_from_labels_raises_on_first_order() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    with pytest.raises(ValueError, match="zero-order consequent layer"):
        model.init_consequents_from_labels(torch.zeros(2, dtype=torch.long))


def test_dgtsk_classifier_prune_structure_empty_features_raises() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="surviving_features must not be empty"):
        model.prune_structure([], [0])


def test_dgtsk_classifier_prune_structure_empty_rules_raises() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="surviving_rules must not be empty"):
        model.prune_structure([0], [])


def test_dgtsk_regressor_prune_structure_empty_features_raises() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    with pytest.raises(ValueError, match="surviving_features must not be empty"):
        model.prune_structure([], [0])


def test_dgtsk_regressor_prune_structure_empty_rules_raises() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    with pytest.raises(ValueError, match="surviving_rules must not be empty"):
        model.prune_structure([0], [])


def test_dgtsk_classifier_search_thresholds_sr_fallback() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.consequent_layer.theta_gates.data.fill_(0.0)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    result = model.search_thresholds(
        x, y, zeta_lambda=[0.0], zeta_theta=[0.0], inplace=True, structural=True, use_lse=False
    )
    assert len(result["surviving_rule_indices"]) == model.n_classes


def test_dgtsk_regressor_search_thresholds_sf_non_empty_no_fallback() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.rule_layer.lambda_gates.data = torch.tensor([1.0, 0.0])
    model.consequent_layer.theta_gates.data.fill_(1.0)
    x = torch.randn(16, 2)
    y = torch.randn(16)
    result = model.search_thresholds(
        x, y, zeta_lambda=[1.0], zeta_theta=[1.0], inplace=True, structural=True, use_lse=False
    )
    assert result["surviving_feature_indices"] == [0]


def test_dgtsk_init_consequents_raises_after_conversion() -> None:
    mfs = {"x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]}
    model_dg = DGTSKClassifierModel(mfs, n_classes=2)
    model_dg.convert_to_first_order()
    with pytest.raises(ValueError, match="requires a zero-order consequent layer"):
        model_dg.init_consequents_from_labels(torch.tensor([0, 1]))


def test_dgtsk_classifier_search_thresholds_no_sr_fallback() -> None:
    mfs = {
        "x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
        "x2": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    }
    model = DGTSKClassifierModel(mfs, n_classes=2, rule_base="cartesian")
    model.consequent_layer.theta_gates.data.copy_(torch.tensor([0.1, 0.5, 0.9, 1.2]))
    x = torch.randn(5, 2)
    y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    result = model.search_thresholds(
        x, y, inplace=True, structural=True, use_lse=False, zeta_lambda=[0.0], zeta_theta=[1.0], verbose=False
    )
    assert len(result["surviving_rule_indices"]) == 3


def test_dgtsk_fit_first_order_consequents_lse_zero_order_raises() -> None:
    # Classifier
    clf = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(5, 3)
    y = torch.randint(0, 2, (5,))
    with pytest.raises(ValueError, match="convert_to_first_order\\(\\) must be called before LSE"):
        clf._fit_first_order_consequents_lse(x, y)

    # Regressor
    reg = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x_reg = torch.randn(5, 2)
    y_reg = torch.randn(5)
    with pytest.raises(ValueError, match="convert_to_first_order\\(\\) must be called before LSE"):
        reg._fit_first_order_consequents_lse(x_reg, y_reg)


def test_dgtsk_search_thresholds_defaults() -> None:
    # Classifier
    clf = DGTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,))
    # Passing zeta_lambda=None and zeta_theta=None, inplace=False, verbose=True
    res_clf = clf.search_thresholds(x, y, zeta_lambda=None, zeta_theta=None, inplace=False, verbose=True)
    assert "best_score" in res_clf

    # Regressor
    reg = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x_reg = torch.randn(10, 2)
    y_reg = torch.randn(10)
    res_reg = reg.search_thresholds(x_reg, y_reg, zeta_lambda=None, zeta_theta=None, inplace=False, verbose=True)
    assert "best_score" in res_reg


def test_dgtsk_search_thresholds_inplace_true_structural_false_use_lse_true() -> None:
    # Classifier
    clf = DGTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,))
    res_clf = clf.search_thresholds(
        x, y, x_val=x, y_val=y, zeta_lambda=[0.5], zeta_theta=[0.5], inplace=True, structural=False, use_lse=True
    )
    assert "best_score" in res_clf

    # Regressor
    reg = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x_reg = torch.randn(10, 2)
    y_reg = torch.randn(10)
    res_reg = reg.search_thresholds(
        x_reg,
        y_reg,
        x_val=x_reg,
        y_val=y_reg,
        zeta_lambda=[0.5],
        zeta_theta=[0.5],
        inplace=True,
        structural=False,
        use_lse=True,
    )
    assert "best_score" in res_reg
