from __future__ import annotations

import pytest
import torch

from highfis.gates import ExpGate
from highfis.layers import (
    AdaptiveDombiRuleLayer,
    ClassificationConsequentLayer,
    GatedClassificationConsequentLayer,
    GatedClassificationZeroOrderConsequentLayer,
    GatedRegressionConsequentLayer,
    GatedRegressionZeroOrderConsequentLayer,
    MembershipLayer,
    RegressionConsequentLayer,
    RuleLayer,
    SparseClassificationConsequentLayer,
    SparseRegressionConsequentLayer,
    _generate_en_frb,
    gate1,
    gate2,
    gate3,
    gate4,
    gate_m,
    resolve_gate_fn,
)
from highfis.memberships import GaussianMF


def _build_input_mfs() -> dict[str, list[GaussianMF]]:
    return {
        "x1": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
        "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    }


def test_membership_layer_forward_shapes() -> None:
    layer = MembershipLayer(_build_input_mfs())
    x = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)

    out = layer(x)

    assert set(out.keys()) == {"x1", "x2"}
    assert out["x1"].shape == (2, 2)
    assert out["x2"].shape == (2, 2)


def test_membership_layer_validates_input_shape() -> None:
    layer = MembershipLayer(_build_input_mfs())
    with pytest.raises(ValueError, match="expected x with 2 dims"):
        layer(torch.tensor([1.0, 2.0]))

    with pytest.raises(ValueError, match="expected 2 inputs"):
        layer(torch.randn(3, 3))


def test_rule_layer_cartesian_forward() -> None:
    m_layer = MembershipLayer(_build_input_mfs())
    r_layer = RuleLayer(["x1", "x2"], [2, 2], rule_base="cartesian", t_norm="prod")
    x = torch.randn(4, 2)

    mu = m_layer(x)
    w = r_layer(mu)

    assert r_layer.n_rules == 4
    assert w.shape == (4, 4)
    assert bool(torch.all(w >= 0.0))


def test_rule_layer_vectorized_forward_matches_manual() -> None:
    m_layer = MembershipLayer(_build_input_mfs())
    r_layer = RuleLayer(["x1", "x2"], [2, 2], rule_base="cartesian", t_norm="prod")
    x = torch.randn(3, 2)

    mu = m_layer(x)
    actual = r_layer(mu)

    expected = []
    for rule in r_layer.rules:
        terms = [mu[name][:, idx] for name, idx in zip(r_layer.input_names, rule, strict=False)]
        expected.append(torch.stack(terms, dim=1).prod(dim=1))

    expected_tensor = torch.stack(expected, dim=1)
    assert torch.allclose(actual, expected_tensor)


def test_rule_layer_registers_rule_indices_buffer() -> None:
    layer = RuleLayer(["x1", "x2"], [2, 2])
    assert isinstance(layer.rule_indices, torch.Tensor)
    assert layer.rule_indices.shape == (4, 2)
    assert layer.rule_indices.dtype == torch.int64


def test_rule_layer_coco_requires_same_mf_count() -> None:
    with pytest.raises(ValueError, match="CoCo rule base"):
        RuleLayer(["x1", "x2"], [2, 3], rule_base="coco")


def test_rule_layer_custom_rejects_invalid_rule_index() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        RuleLayer(["x1", "x2"], [2, 2], rule_base="custom", rules=[(0, 2)])


def test_rule_layer_forward_requires_all_inputs() -> None:
    layer = RuleLayer(["x1", "x2"], [2, 2])
    with pytest.raises(KeyError, match="missing membership output"):
        layer({"x1": torch.rand(3, 2)})


def test_classification_consequent_layer_forward_shape() -> None:
    layer = ClassificationConsequentLayer(n_rules=4, n_inputs=2, n_classes=3)
    x = torch.randn(5, 2)
    norm_w = torch.softmax(torch.randn(5, 4), dim=1)

    logits = layer(x, norm_w)

    assert logits.shape == (5, 3)


def test_gated_classification_consequent_layer_fs_shared_lambda_false() -> None:
    layer = GatedClassificationConsequentLayer(
        n_rules=3,
        n_inputs=2,
        n_classes=2,
        gate_fn=gate1,
        shared_lambda=False,
    )
    layer.mode = "fs"
    x = torch.randn(4, 2)
    norm_w = torch.softmax(torch.randn(4, 3), dim=1)

    logits = layer(x, norm_w)

    assert logits.shape == (4, 2)


def test_gated_classification_consequent_layer_both_shared_lambda_true() -> None:
    layer = GatedClassificationConsequentLayer(
        n_rules=3,
        n_inputs=2,
        n_classes=2,
        gate_fn=gate1,
        shared_lambda=True,
    )
    layer.mode = "both"
    x = torch.randn(4, 2)
    norm_w = torch.softmax(torch.randn(4, 3), dim=1)

    logits = layer(x, norm_w)

    assert logits.shape == (4, 2)


def test_classification_consequent_layer_he_init() -> None:
    """Verify He (Kaiming) initialization on weight and zero bias."""
    layer = ClassificationConsequentLayer(n_rules=4, n_inputs=10, n_classes=3)
    # Bias should be all zeros
    assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
    # Weight std should be close to sqrt(2/fan_in) = sqrt(2/10) ≈ 0.447
    # but can vary; just check it's not the default randn scale (~1.0)
    assert float(layer.weight.detach().std()) < 1.0


def test_sparse_classification_consequent_masks_weights() -> None:
    mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    layer = SparseClassificationConsequentLayer(n_rules=2, n_inputs=2, n_classes=2, rule_feature_mask=mask)
    assert layer.rule_feature_mask.shape == (2, 2)
    assert layer.weight.shape == (2, 2, 2)
    assert layer.bias.shape == (2, 2)

    x = torch.randn(3, 2)
    norm_w = torch.softmax(torch.randn(3, 2), dim=1)
    logits = layer(x, norm_w)

    assert logits.shape == (3, 2)


def test_sparse_regression_consequent_masks_weights() -> None:
    mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    layer = SparseRegressionConsequentLayer(n_rules=2, n_inputs=2, rule_feature_mask=mask)
    assert layer.rule_feature_mask.shape == (2, 2)
    assert layer.weight.shape == (2, 2)
    assert layer.bias.shape == (2,)

    x = torch.randn(3, 2)
    norm_w = torch.softmax(torch.randn(3, 2), dim=1)
    output = layer(x, norm_w)

    assert output.shape == (3, 1)


def test_sparse_classification_consequent_invalid_init_and_forward_shape() -> None:
    mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    with pytest.raises(ValueError, match="n_rules, n_inputs and n_classes must be positive"):
        SparseClassificationConsequentLayer(n_rules=0, n_inputs=2, n_classes=2, rule_feature_mask=mask)

    with pytest.raises(ValueError, match="rule_feature_mask must have shape"):
        SparseClassificationConsequentLayer(
            n_rules=2, n_inputs=2, n_classes=2, rule_feature_mask=torch.ones((1, 2), dtype=torch.bool)
        )

    layer = SparseClassificationConsequentLayer(n_rules=2, n_inputs=2, n_classes=2, rule_feature_mask=mask)
    x = torch.randn(3, 3)
    norm_w = torch.softmax(torch.randn(3, 2), dim=1)
    with pytest.raises(ValueError, match="expected x shape"):
        layer(x, norm_w)

    x = torch.randn(3, 2)
    norm_w = torch.randn(3, 1)
    with pytest.raises(ValueError, match="expected norm_w shape"):
        layer(x, norm_w)


def test_sparse_regression_consequent_invalid_init_and_forward_shape() -> None:
    mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    with pytest.raises(ValueError, match="n_rules and n_inputs must be positive"):
        SparseRegressionConsequentLayer(n_rules=0, n_inputs=2, rule_feature_mask=mask)

    with pytest.raises(ValueError, match="rule_feature_mask must have shape"):
        SparseRegressionConsequentLayer(n_rules=2, n_inputs=2, rule_feature_mask=torch.ones((1, 2), dtype=torch.bool))

    layer = SparseRegressionConsequentLayer(n_rules=2, n_inputs=2, rule_feature_mask=mask)
    x = torch.randn(3, 3)
    norm_w = torch.softmax(torch.randn(3, 2), dim=1)
    with pytest.raises(ValueError, match="expected x shape"):
        layer(x, norm_w)

    x = torch.randn(3, 2)
    norm_w = torch.randn(3, 1)
    with pytest.raises(ValueError, match="expected norm_w shape"):
        layer(x, norm_w)


def test_generate_en_frb_has_unique_rules() -> None:
    rules = _generate_en_frb(3, 2)
    assert len(rules) == len(set(rules))
    assert len(rules) >= 3


# ---------------------------------------------------------------------------
# MembershipLayer validation
# ---------------------------------------------------------------------------


def test_membership_layer_rejects_empty_dict() -> None:
    with pytest.raises(ValueError, match="input_mfs must not be empty"):
        MembershipLayer({})


def test_membership_layer_rejects_empty_mf_list() -> None:
    with pytest.raises(ValueError, match="must define at least one membership function"):
        MembershipLayer({"x1": []})


# ---------------------------------------------------------------------------
# RuleLayer validation
# ---------------------------------------------------------------------------


def test_rule_layer_rejects_empty_input_names() -> None:
    with pytest.raises(ValueError, match="input_names must not be empty"):
        RuleLayer([], [])


def test_rule_layer_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        RuleLayer(["x1"], [2, 3])


def test_rule_layer_rejects_invalid_rule_base() -> None:
    with pytest.raises(ValueError, match="rule_base must be"):
        RuleLayer(["x1"], [2], rule_base="invalid")


def test_rule_layer_coco_happy_path() -> None:
    """CoCo rule base with equal MF counts covers lines 119-123."""
    mfs = {
        "x1": [GaussianMF(mean=float(i), sigma=1.0) for i in range(3)],
        "x2": [GaussianMF(mean=float(i), sigma=1.0) for i in range(3)],
    }
    layer = RuleLayer(["x1", "x2"], [3, 3], rule_base="coco")
    assert layer.n_rules == 3
    m = MembershipLayer(mfs)
    x = torch.randn(4, 2)
    w = layer(m(x))
    assert w.shape == (4, 3)


def test_rule_layer_en_single_input_triggers_seen_collision() -> None:
    """En-FRB with s=2, d=1 triggers base_t-in-seen branch (22->26) and line 130."""
    mfs = {"x1": [GaussianMF(mean=float(i), sigma=1.0) for i in range(2)]}
    layer = RuleLayer(["x1"], [2], rule_base="en")
    assert layer.n_rules == 2
    m = MembershipLayer(mfs)
    x = torch.randn(4, 1)
    w = layer(m(x))
    assert w.shape == (4, 2)


def test_rule_layer_en_rejects_unequal_mf_counts() -> None:
    with pytest.raises(ValueError, match="En-FRB"):
        RuleLayer(["x1", "x2"], [2, 3], rule_base="en")


def test_rule_layer_custom_rejects_wrong_rule_size() -> None:
    with pytest.raises(ValueError, match="has size"):
        RuleLayer(["x1", "x2"], [2, 2], rule_base="custom", rules=[(0,)])


def test_rule_layer_custom_rejects_empty_rules() -> None:
    with pytest.raises(ValueError, match="rules must not be empty"):
        RuleLayer(["x1", "x2"], [2, 2], rule_base="custom", rules=[])


def test_rule_layer_custom_rejects_missing_rules() -> None:
    with pytest.raises(ValueError, match="rules must be provided when rule_base='custom'"):
        RuleLayer(["x1", "x2"], [2, 2], rule_base="custom")


def test_rule_layer_custom_t_norm_fn_overrides_default() -> None:
    """Custom callable passed via t_norm uses the custom function."""
    layer = RuleLayer(
        ["x1", "x2"],
        [2, 2],
        t_norm=lambda t: t.prod(dim=-1),
    )
    m = MembershipLayer(_build_input_mfs())
    x = torch.randn(3, 2)
    w = layer(m(x))

    mu = m(x)
    expected = []
    for rule in layer.rules:
        terms = [mu[name][:, idx] for name, idx in zip(layer.input_names, rule, strict=False)]
        expected.append(torch.stack(terms, dim=1).prod(dim=1))
    expected_tensor = torch.stack(expected, dim=1)

    assert w.shape == expected_tensor.shape
    assert torch.allclose(w, expected_tensor)


def test_rule_layer_supports_fuco_alias() -> None:
    layer = RuleLayer(["x1", "x2"], [2, 2], rule_base="fuco")
    x = torch.randn(3, 2)
    m = MembershipLayer(_build_input_mfs())
    w = layer(m(x))

    cartesian = RuleLayer(["x1", "x2"], [2, 2], rule_base="cartesian")
    expected = cartesian(m(x))

    assert w.shape == (3, 4)
    assert torch.allclose(w, expected)
    w = layer(m(torch.randn(4, 2)))
    assert w.shape == (4, layer.n_rules)


def test_adaptive_dombi_rule_layer_forward_matches_manual() -> None:
    m_layer = MembershipLayer(_build_input_mfs())
    r_layer = AdaptiveDombiRuleLayer(["x1", "x2"], [2, 2], rule_base="cartesian", lambda_init=1.0)
    x = torch.rand(4, 2)

    mu = m_layer(x)
    actual = r_layer(mu)

    expected = []
    for rule_idx, rule in enumerate(r_layer.rules):
        terms = [mu[name][:, idx] for name, idx in zip(r_layer.input_names, rule, strict=False)]
        tensor_terms = torch.stack(terms, dim=1).clamp(min=r_layer.eps, max=1.0 - r_layer.eps)
        ratio = (1.0 - tensor_terms) / tensor_terms
        lam = r_layer.lambdas[rule_idx]
        sum_ratio = ratio.pow(lam).sum(dim=-1)
        expected.append((1.0 + sum_ratio).pow(-1.0 / lam))

    expected_tensor = torch.stack(expected, dim=1)
    assert torch.allclose(actual, expected_tensor)


# ---------------------------------------------------------------------------
# ClassificationConsequentLayer validation
# ---------------------------------------------------------------------------


def test_classification_consequent_layer_rejects_bad_x_shape() -> None:
    layer = ClassificationConsequentLayer(n_rules=4, n_inputs=2, n_classes=3)
    norm_w = torch.softmax(torch.randn(5, 4), dim=1)
    with pytest.raises(ValueError, match="expected x shape"):
        layer(torch.randn(5, 3), norm_w)


def test_classification_consequent_layer_rejects_bad_normw_shape() -> None:
    layer = ClassificationConsequentLayer(n_rules=4, n_inputs=2, n_classes=3)
    x = torch.randn(5, 2)
    with pytest.raises(ValueError, match="expected norm_w shape"):
        layer(x, torch.randn(5, 3))


def test_dga_ltsk_rule_layer_invalid_alpha_init_raises() -> None:
    from highfis.layers import DGALETSKRuleLayer

    with pytest.raises(ValueError, match="alpha_init must be > 0"):
        DGALETSKRuleLayer(["x1", "x2"], [2, 2], alpha_init=0.0)


def test_dga_ltsk_rule_layer_missing_membership_output_raises() -> None:
    from highfis.layers import DGALETSKRuleLayer

    layer = DGALETSKRuleLayer(["x1", "x2"], [2, 2])
    with pytest.raises(KeyError, match="missing membership output"):
        layer({"x1": torch.rand(2, 2)})


def test_dgtsk_rule_layer_missing_membership_output_raises() -> None:
    from highfis.layers import DGTSKRuleLayer

    layer = DGTSKRuleLayer(["x1", "x2"], [2, 2])
    with pytest.raises(KeyError, match="missing membership output"):
        layer({"x1": torch.rand(2, 2)})


def test_adasoftmin_rule_layer_missing_membership_output_raises() -> None:
    from highfis.layers import AdaSoftminRuleLayer

    layer = AdaSoftminRuleLayer(["x1", "x2"], [2, 2])
    with pytest.raises(KeyError, match="missing membership output"):
        layer({"x1": torch.rand(2, 2)})


def test_adp_softmin_rule_layer_missing_membership_output_raises() -> None:
    from highfis.layers import ADPSoftminRuleLayer

    layer = ADPSoftminRuleLayer(["x1", "x2"], [2, 2])
    with pytest.raises(KeyError, match="missing membership output"):
        layer({"x1": torch.rand(2, 2)})


def test_adp_softmin_rule_layer_rejects_nonpositive_kappa() -> None:
    from highfis.layers import ADPSoftminRuleLayer

    with pytest.raises(ValueError, match="kappa must be > 0"):
        ADPSoftminRuleLayer(["x1"], [2], kappa=0.0)


def test_adp_softmin_rule_layer_rejects_nonpositive_xi() -> None:
    from highfis.layers import ADPSoftminRuleLayer

    with pytest.raises(ValueError, match="xi must be > 0"):
        ADPSoftminRuleLayer(["x1"], [2], xi=0.0)


def test_adp_softmin_rule_layer_forward_shape() -> None:
    from highfis.layers import ADPSoftminRuleLayer

    layer = ADPSoftminRuleLayer(["x1", "x2"], [2, 2])
    membership_outputs = {
        "x1": torch.tensor([[0.4, 0.8], [0.2, 0.9]]),
        "x2": torch.tensor([[0.7, 0.3], [0.5, 0.4]]),
    }
    output = layer(membership_outputs)

    assert output.shape == (2, 4)
    assert torch.all(output > 0.0)
    assert torch.all(output < 1.0)


def test_classification_consequent_layer_invalid_init_args() -> None:
    with pytest.raises(ValueError, match="n_rules, n_inputs and n_classes must be positive"):
        ClassificationConsequentLayer(n_rules=0, n_inputs=2, n_classes=2)


def test_adaptive_dombi_rule_layer_rejects_nonpositive_lambda_init() -> None:
    with pytest.raises(ValueError, match="lambda_init must be > 0"):
        AdaptiveDombiRuleLayer(["x1", "x2"], [2, 2], lambda_init=0.0)


def test_adaptive_dombi_rule_layer_missing_membership_output_raises() -> None:
    layer = AdaptiveDombiRuleLayer(["x1", "x2"], [2, 2])
    with pytest.raises(KeyError, match="missing membership output"):
        layer({"x1": torch.rand(2, 2)})


# ---------------------------------------------------------------------------
# RegressionConsequentLayer
# ---------------------------------------------------------------------------


def test_regression_consequent_layer_forward_shape() -> None:
    layer = RegressionConsequentLayer(n_rules=4, n_inputs=2)
    x = torch.randn(5, 2)
    norm_w = torch.softmax(torch.randn(5, 4), dim=1)

    out = layer(x, norm_w)

    assert out.shape == (5, 1)


def test_regression_consequent_layer_he_init() -> None:
    """Verify He (Kaiming) initialization on weight and zero bias."""
    layer = RegressionConsequentLayer(n_rules=4, n_inputs=10)
    assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
    assert float(layer.weight.detach().std()) < 1.0


def test_regression_consequent_layer_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="n_rules and n_inputs must be positive"):
        RegressionConsequentLayer(n_rules=0, n_inputs=2)
    with pytest.raises(ValueError, match="n_rules and n_inputs must be positive"):
        RegressionConsequentLayer(n_rules=4, n_inputs=0)


def test_regression_consequent_layer_rejects_bad_x_shape() -> None:
    layer = RegressionConsequentLayer(n_rules=4, n_inputs=2)
    norm_w = torch.softmax(torch.randn(5, 4), dim=1)
    with pytest.raises(ValueError, match="expected x shape"):
        layer(torch.randn(5, 3), norm_w)


def test_regression_consequent_layer_rejects_bad_normw_shape() -> None:
    layer = RegressionConsequentLayer(n_rules=4, n_inputs=2)
    x = torch.randn(5, 2)
    with pytest.raises(ValueError, match="expected norm_w shape"):
        layer(x, torch.randn(5, 3))


def test_regression_consequent_layer_gradient_flows() -> None:
    """Verify that gradients flow through the regression consequent layer."""
    layer = RegressionConsequentLayer(n_rules=4, n_inputs=2)
    x = torch.randn(5, 2, requires_grad=True)
    norm_w = torch.softmax(torch.randn(5, 4), dim=1)

    out = layer(x, norm_w)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_gate_functions_and_resolver() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(gate1(x), torch.sigmoid(x))
    assert torch.allclose(gate2(x), 1.0 - torch.exp(-x.pow(2)))
    assert torch.allclose(gate3(x), torch.exp(-x.pow(2)))
    assert torch.allclose(gate4(x), x * torch.sqrt(torch.exp(1.0 - x.pow(2))))
    assert torch.allclose(gate_m(x), x.pow(2) * torch.exp(1.0 - x.pow(2)))
    assert resolve_gate_fn("gate1") is gate1
    default = resolve_gate_fn(None)
    assert isinstance(default, ExpGate)
    assert default.k == 10.0
    assert resolve_gate_fn(torch.sigmoid) is torch.sigmoid
    with pytest.raises(ValueError, match="unsupported gate function"):
        resolve_gate_fn("invalid")


def test_gated_classification_consequent_layer_forward_shape() -> None:
    layer = GatedClassificationConsequentLayer(n_rules=4, n_inputs=2, n_classes=2, gate_fn="gate1")
    x = torch.randn(5, 2)
    norm_w = torch.softmax(torch.randn(5, 4), dim=1)

    out = layer(x, norm_w)
    assert out.shape == (5, 2)


def test_gated_classification_zero_order_consequent_layer_forward_shape() -> None:
    layer = GatedClassificationZeroOrderConsequentLayer(n_rules=4, n_inputs=2, n_classes=2, gate_fn="gate2")
    x = torch.randn(5, 2)
    norm_w = torch.softmax(torch.randn(5, 4), dim=1)

    out = layer(x, norm_w)
    assert out.shape == (5, 2)


def test_gated_regression_consequent_layer_forward_shape() -> None:
    layer = GatedRegressionConsequentLayer(n_rules=4, n_inputs=2, gate_fn="gate3")
    x = torch.randn(5, 2)
    norm_w = torch.softmax(torch.randn(5, 4), dim=1)

    out = layer(x, norm_w)
    assert out.shape == (5, 1)


def test_gated_regression_zero_order_consequent_layer_forward_shape() -> None:
    layer = GatedRegressionZeroOrderConsequentLayer(n_rules=4, n_inputs=2, gate_fn="gate4")
    x = torch.randn(5, 2)
    norm_w = torch.softmax(torch.randn(5, 4), dim=1)

    out = layer(x, norm_w)
    assert out.shape == (5, 1)


def test_gated_consequent_layer_rejects_bad_shapes() -> None:
    layer = GatedClassificationConsequentLayer(n_rules=4, n_inputs=2, n_classes=2)
    with pytest.raises(ValueError, match="expected x shape"):
        layer(torch.randn(5, 3), torch.softmax(torch.randn(5, 4), dim=1))
    with pytest.raises(ValueError, match="expected norm_w shape"):
        layer(torch.randn(5, 2), torch.randn(5, 3))


def test_gated_classification_zero_order_consequent_layer_rejects_bad_shapes() -> None:
    layer = GatedClassificationZeroOrderConsequentLayer(n_rules=4, n_inputs=2, n_classes=2)
    with pytest.raises(ValueError, match="expected x shape"):
        layer(torch.randn(5, 3), torch.softmax(torch.randn(5, 4), dim=1))
    with pytest.raises(ValueError, match="expected norm_w shape"):
        layer(torch.randn(5, 2), torch.randn(5, 3))


def test_gated_regression_zero_order_consequent_layer_rejects_bad_shapes() -> None:
    layer = GatedRegressionZeroOrderConsequentLayer(n_rules=4, n_inputs=2)
    with pytest.raises(ValueError, match="expected x shape"):
        layer(torch.randn(5, 3), torch.softmax(torch.randn(5, 4), dim=1))
    with pytest.raises(ValueError, match="expected norm_w shape"):
        layer(torch.randn(5, 2), torch.randn(5, 3))


def test_gated_regression_consequent_layer_rejects_bad_shapes() -> None:
    layer = GatedRegressionConsequentLayer(n_rules=4, n_inputs=2)
    with pytest.raises(ValueError, match="expected x shape"):
        layer(torch.randn(5, 3), torch.softmax(torch.randn(5, 4), dim=1))
    with pytest.raises(ValueError, match="expected norm_w shape"):
        layer(torch.randn(5, 2), torch.randn(5, 3))


def test_gated_consequent_layer_invalid_init_args() -> None:
    with pytest.raises(ValueError, match="n_rules, n_inputs and n_classes must be positive"):
        GatedClassificationConsequentLayer(n_rules=0, n_inputs=2, n_classes=2)
    with pytest.raises(ValueError, match="n_rules, n_inputs and n_classes must be positive"):
        GatedClassificationZeroOrderConsequentLayer(n_rules=0, n_inputs=2, n_classes=2)
    with pytest.raises(ValueError, match="n_rules and n_inputs must be positive"):
        GatedRegressionZeroOrderConsequentLayer(n_rules=0, n_inputs=2)
    with pytest.raises(ValueError, match="n_rules and n_inputs must be positive"):
        GatedRegressionConsequentLayer(n_rules=0, n_inputs=2)
