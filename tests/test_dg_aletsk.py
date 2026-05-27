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
    # tau = max_gate - 0.5 * (max_gate - min_gate)
    fv = model.get_feature_gate_values().detach()
    rv = model.get_rule_gate_values().detach()
    expected_tau_lambda = float(fv.max()) - 0.5 * float(fv.max() - fv.min())
    expected_tau_theta = float(rv.max()) - 0.5 * float(rv.max() - rv.min())
    assert abs(tau_lambda - expected_tau_lambda) < 1e-5
    assert abs(tau_theta - expected_tau_theta) < 1e-5

    model.apply_thresholds(tau_lambda, tau_theta)
    assert model.rule_layer.lambda_gates.data[0] == 1.0
    assert model.consequent_layer.theta_gates.data[0] == 1.0


def test_dgaletsk_search_thresholds_returns_result() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=5, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(x, y, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], inplace=False)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}
    assert 0.0 <= result["best_score"] <= 1.0


def test_dgaletsk_classifier_invalid_zeta_raises() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match=r"zeta must be in \[0, 1\]"):
        model.compute_thresholds(-0.1, 0.5)


def test_dgaletsk_classifier_apply_thresholds_invalid_thresholds_raises() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="thresholds must be finite"):
        model.apply_thresholds(float("nan"), 0.0)


def test_dgaletsk_classifier_search_thresholds_with_lse_returns_result() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=5, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        use_lse=True,
        inplace=False,
    )

    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}
    assert 0.0 <= result["best_score"] <= 1.0


def test_dgaletsk_classifier_search_thresholds_with_validation_and_verbose() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=5, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        x_val=x,
        y_val=y,
        use_lse=False,
        inplace=True,
        verbose=True,
    )

    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}
    assert 0.0 <= result["best_score"] <= 1.0


def test_dgaletsk_regressor_search_thresholds_with_lse_returns_result() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)
    model.fit_dg_phase(x, y, epochs=5, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        use_lse=True,
        inplace=False,
    )

    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgaletsk_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DGALETSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_dgaletsk_rule_layer_lambda_gates_shape_is_per_feature() -> None:
    """lambda_gates must be (n_inputs,) — shared across all rules per the DG-ALETSK paper."""
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=4, n_mfs=2), n_classes=2)
    assert model.rule_layer.lambda_gates.shape == (4,)


def test_dgaletsk_regressor_lambda_gates_shape_is_per_feature() -> None:
    """lambda_gates must be (n_inputs,) — shared across all rules per the DG-ALETSK paper."""
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=5, n_mfs=2))
    assert model.rule_layer.lambda_gates.shape == (5,)


def test_dgaletsk_classifier_first_order_consequent_mode_is_re() -> None:
    """After convert_to_first_order(), consequent mode must be 're' (rule gates only)."""
    model = DGALETSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    assert model.consequent_layer.mode == "re"


def test_dgaletsk_regressor_first_order_consequent_mode_is_re() -> None:
    """After convert_to_first_order(), consequent mode must be 're' (rule gates only)."""
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    assert model.consequent_layer.mode == "re"


def test_dgaletsk_rule_layer_firing_strengths_in_unit_interval() -> None:
    """ALE softmin converges to min(mu_d) per paper eq. (22) with adaptive q̂."""

    from highfis.layers import DGALETSKRuleLayer

    layer = DGALETSKRuleLayer(
        input_names=["x1", "x2", "x3"],
        mf_per_input=[2, 2, 2],
        rule_base="coco",
    )
    # gate4(1.0) = 1.0 * sqrt(exp(0)) = 1.0 — feature gates exactly 1
    layer.lambda_gates.data.fill_(1.0)

    # CoCo-FRB with 2 MFs creates 2 rules; 3 inputs
    # Rule 0 activations: mu = [0.8, 0.5, 0.3]  → min = 0.3
    # Rule 1 activations: mu = [0.9, 0.7, 0.6]  → min = 0.6
    mf_outputs = {
        "x1": torch.tensor([[0.8, 0.9]]),
        "x2": torch.tensor([[0.5, 0.7]]),
        "x3": torch.tensor([[0.3, 0.6]]),
    }
    with torch.no_grad():
        f = layer(mf_outputs)

    assert f.shape == (1, 2)
    assert torch.allclose(f[0, 0], torch.tensor(0.3), atol=0.01), f"expected ≈0.3, got {f[0, 0].item():.4f}"
    assert torch.allclose(f[0, 1], torch.tensor(0.6), atol=0.01), f"expected ≈0.6, got {f[0, 1].item():.4f}"
