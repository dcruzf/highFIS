from __future__ import annotations

import pytest
import torch

from highfis.layers import (
    ClassificationConsequentLayer,
    MembershipLayer,
    NormalizationLayer,
    RegressionConsequentLayer,
    RuleLayer,
    _generate_en_frb,
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


def test_normalization_layer_normalizes_rows() -> None:
    layer = NormalizationLayer()
    w = torch.tensor([[1.0, 1.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    norm = layer(w)

    assert norm.shape == w.shape
    assert torch.allclose(norm.sum(dim=1), torch.ones(2), atol=1e-6)


def test_normalization_layer_rejects_non_matrix() -> None:
    layer = NormalizationLayer()
    with pytest.raises(ValueError, match="expected w with 2 dims"):
        layer(torch.rand(2, 2, 2))


def test_classification_consequent_layer_forward_shape() -> None:
    layer = ClassificationConsequentLayer(n_rules=4, n_inputs=2, n_classes=3)
    x = torch.randn(5, 2)
    norm_w = torch.softmax(torch.randn(5, 4), dim=1)

    logits = layer(x, norm_w)

    assert logits.shape == (5, 3)


def test_classification_consequent_layer_he_init() -> None:
    """Verify He (Kaiming) initialization on weight and zero bias."""
    layer = ClassificationConsequentLayer(n_rules=4, n_inputs=10, n_classes=3)
    # Bias should be all zeros
    assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
    # Weight std should be close to sqrt(2/fan_in) = sqrt(2/10) ≈ 0.447
    # but can vary; just check it's not the default randn scale (~1.0)
    assert float(layer.weight.detach().std()) < 1.0


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


def test_rule_layer_custom_t_norm_fn_overrides_default() -> None:
    """t_norm_fn parameter uses the custom function (line 195)."""
    layer = RuleLayer(
        ["x1", "x2"],
        [2, 2],
        t_norm_fn=lambda t: t.prod(dim=-1),
    )
    m = MembershipLayer(_build_input_mfs())
    w = layer(m(torch.randn(4, 2)))
    assert w.shape == (4, layer.n_rules)


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
