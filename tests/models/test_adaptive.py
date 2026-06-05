from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from highfis.layers import AdaSoftminRuleLayer
from highfis.memberships import ADATSKGaussianMF, CompositeGaussianMF, GaussianMF
from highfis.models import (
    ADATSKClassifierModel,
    ADATSKRegressorModel,
    ADMTSKClassifierModel,
    ADMTSKRegressorModel,
    ADPTSKClassifierModel,
    ADPTSKRegressorModel,
)
from highfis.t_norms import DombiTNorm


def _build_adatsk_input_mfs(n_inputs: int = 3, n_rules: int = 2) -> dict[str, list[CompositeGaussianMF]]:
    return {
        f"x{i + 1}": [CompositeGaussianMF(mean=float(j), sigma=1.0, eps=1e-4) for j in range(n_rules)]
        for i in range(n_inputs)
    }


def test_adatsk_classifier_forward_predict_shapes() -> None:
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_adatsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)
    x = torch.randn(6, 2)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_adatsk_regressor_forward_shape() -> None:
    model = ADATSKRegressorModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2))
    x = torch.randn(5, 2)

    output = model.forward(x)

    assert output.shape == (5, 1)


def test_adatsk_classifier_fit_returns_history() -> None:
    torch.manual_seed(0)
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,), dtype=torch.long)

    history = model.fit(x, y, epochs=3, learning_rate=1e-2, batch_size=5)

    assert set(history.keys()) == {"train", "ur", "val", "stopped_epoch"}
    assert len(history["train"]) == 3
    assert len(history["ur"]) == 3
    assert history["stopped_epoch"] == 3


def test_adatsk_classifier_uses_ada_softmin_rule_layer() -> None:
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(), n_classes=2)

    assert isinstance(model.rule_layer, AdaSoftminRuleLayer)


def test_adatsk_regressor_uses_ada_softmin_rule_layer() -> None:
    model = ADATSKRegressorModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2))

    assert isinstance(model.rule_layer, AdaSoftminRuleLayer)


def test_adatsk_classifier_criterion_is_mse() -> None:
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)

    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_adatsk_classifier_default_optimizer_is_sgd() -> None:
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)

    optimizer = model._build_optimizer(None, learning_rate=1e-2, weight_decay=0.0)

    assert isinstance(optimizer, torch.optim.SGD)


def test_adatsk_gaussian_mf_matches_paper_eq3_when_sigma_one() -> None:
    mf = ADATSKGaussianMF(mean=0.0, sigma=1.0)
    x = torch.tensor([-1.0, 0.0, 2.0])

    values = mf(x)
    expected = torch.exp(-x.square())

    assert torch.allclose(values, expected, atol=1e-6)


def test_adatsk_classifier_consequents_are_zero_initialized() -> None:
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)

    weight = cast(torch.Tensor, model.consequent_layer.weight).detach()
    bias = cast(torch.Tensor, model.consequent_layer.bias).detach()

    assert torch.allclose(weight, torch.zeros_like(weight))
    assert torch.allclose(bias, torch.zeros_like(bias))


def test_adatsk_classifier_model_optimizer_uses_custom_instance() -> None:
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)
    custom = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer = model._build_optimizer(custom, learning_rate=1e-2, weight_decay=0.0)

    assert optimizer is custom


def test_adatsk_regressor_model_optimizer_uses_custom_instance() -> None:
    model = ADATSKRegressorModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2))
    custom = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer = model._build_optimizer(custom, learning_rate=1e-2, weight_decay=0.0)

    assert optimizer is custom


def test_adatsk_classifier_can_disable_zero_init() -> None:
    torch.manual_seed(0)
    model = ADATSKClassifierModel(
        _build_adatsk_input_mfs(n_inputs=2, n_rules=2),
        n_classes=2,
        zero_consequent_init=False,
    )

    weight = cast(torch.Tensor, model.consequent_layer.weight).detach()
    assert not torch.allclose(weight, torch.zeros_like(weight))


def test_adatsk_classifier_zero_init_skips_non_tensor_weight() -> None:
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)

    class _BiasOnlyConsequent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = object()
            self.bias = torch.ones(2, 2)

    fake = _BiasOnlyConsequent()
    model.consequent_layer = cast(nn.Module, fake)

    model._zero_initialize_consequents()

    assert torch.allclose(fake.bias, torch.zeros_like(fake.bias))


def test_adatsk_classifier_zero_init_skips_non_tensor_bias() -> None:
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)

    class _WeightOnlyConsequent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.ones(2, 2, 2)
            self.bias = object()

    fake = _WeightOnlyConsequent()
    model.consequent_layer = cast(nn.Module, fake)

    model._zero_initialize_consequents()

    assert torch.allclose(fake.weight, torch.zeros_like(fake.weight))


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_adptsk_classifier_forward_predict_shapes() -> None:
    model = ADPTSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_adptsk_classifier_validates_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        ADPTSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_adptsk_classifier_validates_kappa_xi() -> None:
    with pytest.raises(ValueError, match="kappa must be > 0"):
        ADPTSKClassifierModel(_build_input_mfs(), n_classes=2, kappa=0.0)
    with pytest.raises(ValueError, match="xi must be > 0"):
        ADPTSKClassifierModel(_build_input_mfs(), n_classes=2, xi=-1.0)


def test_adptsk_regressor_validates_kappa_xi() -> None:
    with pytest.raises(ValueError, match="kappa must be > 0"):
        ADPTSKRegressorModel(_build_input_mfs(), kappa=-5.0)
    with pytest.raises(ValueError, match="xi must be > 0"):
        ADPTSKRegressorModel(_build_input_mfs(), xi=0.0)


def test_adptsk_regressor_forward_predict_shape() -> None:
    model = ADPTSKRegressorModel(_build_input_mfs(), rule_base="coco")
    x = torch.randn(6, 3)

    output = model.forward(x)
    pred = model.predict(x)

    assert output.shape == (6, 1)
    assert pred.shape == (6,)


def test_adptsk_classifier_default_criterion_is_mse() -> None:
    model = ADPTSKClassifierModel(_build_input_mfs(), n_classes=2)
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_adptsk_classifier_default_optimizer_is_adam() -> None:
    model = ADPTSKClassifierModel(_build_input_mfs(), n_classes=2)
    optimizer = model._build_optimizer(None, learning_rate=1e-3, weight_decay=0.0)
    assert isinstance(optimizer, torch.optim.Adam)


def test_adptsk_classifier_optimizer_passthrough() -> None:
    model = ADPTSKClassifierModel(_build_input_mfs(), n_classes=2)
    provided = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer = model._build_optimizer(provided, learning_rate=1e-3, weight_decay=0.0)
    assert optimizer is provided


def test_adptsk_classifier_optimizer_with_consequent_batch_norm_uses_adam() -> None:
    model = ADPTSKClassifierModel(_build_input_mfs(), n_classes=2, consequent_batch_norm=True)
    optimizer = model._build_optimizer(None, learning_rate=1e-3, weight_decay=0.0)
    assert isinstance(optimizer, torch.optim.Adam)


def test_adptsk_regressor_default_optimizer_is_adam() -> None:
    model = ADPTSKRegressorModel(_build_input_mfs(), rule_base="coco")
    optimizer = model._build_optimizer(None, learning_rate=1e-3, weight_decay=0.0)
    assert isinstance(optimizer, torch.optim.Adam)


def test_adptsk_regressor_optimizer_passthrough() -> None:
    model = ADPTSKRegressorModel(_build_input_mfs(), rule_base="coco")
    provided = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer = model._build_optimizer(provided, learning_rate=1e-3, weight_decay=0.0)
    assert optimizer is provided


def test_adptsk_regressor_optimizer_with_consequent_batch_norm_uses_adam() -> None:
    model = ADPTSKRegressorModel(_build_input_mfs(), rule_base="coco", consequent_batch_norm=True)
    optimizer = model._build_optimizer(None, learning_rate=1e-3, weight_decay=0.0)
    assert isinstance(optimizer, torch.optim.Adam)


def test_adptsk_classifier_zero_initializes_consequents_by_default() -> None:
    model = ADPTSKClassifierModel(_build_input_mfs(), n_classes=2)
    weight = getattr(model.consequent_layer, "weight", None)
    bias = getattr(model.consequent_layer, "bias", None)
    assert isinstance(weight, torch.Tensor)
    assert isinstance(bias, torch.Tensor)
    assert torch.allclose(weight, torch.zeros_like(weight))
    assert torch.allclose(bias, torch.zeros_like(bias))


def test_adptsk_classifier_can_disable_zero_consequent_init() -> None:
    model = ADPTSKClassifierModel(_build_input_mfs(), n_classes=2, zero_consequent_init=False)
    weight = getattr(model.consequent_layer, "weight", None)
    assert isinstance(weight, torch.Tensor)
    assert not torch.allclose(weight, torch.zeros_like(weight))


def test_adptsk_regressor_can_disable_zero_consequent_init() -> None:
    model = ADPTSKRegressorModel(_build_input_mfs(), rule_base="coco", zero_consequent_init=False)
    weight = getattr(model.consequent_layer, "weight", None)
    assert isinstance(weight, torch.Tensor)
    assert not torch.allclose(weight, torch.zeros_like(weight))


def test_adptsk_classifier_zero_initialize_handles_missing_params() -> None:
    model = ADPTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.consequent_layer = cast(nn.Module, nn.Identity())
    model._zero_initialize_consequents()


def test_adptsk_regressor_zero_initialize_handles_missing_params() -> None:
    model = ADPTSKRegressorModel(_build_input_mfs(), rule_base="coco")
    model.consequent_layer = cast(nn.Module, nn.Identity())
    model._zero_initialize_consequents()


def test_admtsk_classifier_forward_predict_shapes() -> None:
    model = ADMTSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_admtsk_classifier_default_criterion_is_mse() -> None:
    model = ADMTSKClassifierModel(_build_input_mfs(), n_classes=2)
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_admtsk_classifier_default_optimizer_is_adam() -> None:
    model = ADMTSKClassifierModel(_build_input_mfs(), n_classes=2)
    optimizer = model._build_optimizer(None, learning_rate=1e-2, weight_decay=0.0)
    assert isinstance(optimizer, torch.optim.Adam)


def test_admtsk_classifier_optimizer_returns_custom_optimizer() -> None:
    model = ADMTSKClassifierModel(_build_input_mfs(), n_classes=2)
    custom = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer = model._build_optimizer(custom, learning_rate=1e-2, weight_decay=0.0)
    assert optimizer is custom


def test_admtsk_classifier_optimizer_includes_bn_params_when_enabled() -> None:
    model = ADMTSKClassifierModel(_build_input_mfs(), n_classes=2, consequent_batch_norm=True)
    optimizer = model._build_optimizer(None, learning_rate=1e-2, weight_decay=0.0)
    cons_only = len(list(model.consequent_layer.parameters()))
    cons_group = optimizer.param_groups[2]["params"]
    assert len(cons_group) > cons_only


def test_admtsk_classifier_zero_initializes_consequents_by_default() -> None:
    model = ADMTSKClassifierModel(_build_input_mfs(), n_classes=2)
    weight = getattr(model.consequent_layer, "weight", None)
    bias = getattr(model.consequent_layer, "bias", None)
    assert isinstance(weight, torch.Tensor)
    assert isinstance(bias, torch.Tensor)
    assert torch.allclose(weight, torch.zeros_like(weight))
    assert torch.allclose(bias, torch.zeros_like(bias))


def test_admtsk_classifier_can_disable_zero_consequent_init() -> None:
    model = ADMTSKClassifierModel(
        _build_input_mfs(),
        n_classes=2,
        zero_consequent_init=False,
    )
    weight = getattr(model.consequent_layer, "weight", None)
    assert isinstance(weight, torch.Tensor)
    assert not torch.allclose(weight, torch.zeros_like(weight))


def test_admtsk_regressor_forward_shape() -> None:
    model = ADMTSKRegressorModel(_build_input_mfs(), rule_base="coco")
    x = torch.randn(5, 3)

    output = model.forward(x)
    pred = model.predict(x)

    assert output.shape == (5, 1)
    assert pred.shape == (5,)


def test_admtsk_regressor_default_optimizer_is_adam() -> None:
    model = ADMTSKRegressorModel(_build_input_mfs(), rule_base="coco")
    optimizer = model._build_optimizer(None, learning_rate=1e-2, weight_decay=0.0)
    assert isinstance(optimizer, torch.optim.Adam)


def test_admtsk_regressor_optimizer_returns_custom_optimizer() -> None:
    model = ADMTSKRegressorModel(_build_input_mfs(), rule_base="coco")
    custom = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer = model._build_optimizer(custom, learning_rate=1e-2, weight_decay=0.0)
    assert optimizer is custom


def test_admtsk_regressor_optimizer_includes_bn_params_when_enabled() -> None:
    model = ADMTSKRegressorModel(_build_input_mfs(), rule_base="coco", consequent_batch_norm=True)
    optimizer = model._build_optimizer(None, learning_rate=1e-2, weight_decay=0.0)
    cons_only = len(list(model.consequent_layer.parameters()))
    cons_group = optimizer.param_groups[2]["params"]
    assert len(cons_group) > cons_only


def test_admtsk_regressor_can_disable_zero_consequent_init() -> None:
    model = ADMTSKRegressorModel(
        _build_input_mfs(),
        rule_base="coco",
        zero_consequent_init=False,
    )
    weight = getattr(model.consequent_layer, "weight", None)
    assert isinstance(weight, torch.Tensor)
    assert not torch.allclose(weight, torch.zeros_like(weight))


def test_admtsk_classifier_zero_init_noop_when_weight_and_bias_not_tensors() -> None:
    class FakeConsequent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Identity()
            self.bias = nn.Identity()

        def forward(self, x: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
            return x.new_zeros((x.shape[0], 2))

    model = ADMTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.consequent_layer = FakeConsequent()

    model._zero_initialize_consequents()

    assert isinstance(model.consequent_layer.weight, nn.Module)
    assert isinstance(model.consequent_layer.bias, nn.Module)


def test_admtsk_regressor_zero_init_noop_when_weight_and_bias_not_tensors() -> None:
    class FakeConsequent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Identity()
            self.bias = nn.Identity()

        def forward(self, x: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
            return x.new_zeros((x.shape[0], 1))

    model = ADMTSKRegressorModel(_build_input_mfs(), rule_base="coco")
    model.consequent_layer = FakeConsequent()

    model._zero_initialize_consequents()

    assert isinstance(model.consequent_layer.weight, nn.Module)
    assert isinstance(model.consequent_layer.bias, nn.Module)


def test_admtsk_classifier_fixed_lambda_branch() -> None:
    model = ADMTSKClassifierModel(_build_input_mfs(), n_classes=2, adaptive=False, lambda_=2.0)
    x = torch.randn(6, 3)
    out = model.forward(x)
    assert out.shape == (6, 2)


def test_admtsk_regressor_fixed_lambda_branch() -> None:
    model = ADMTSKRegressorModel(_build_input_mfs(), adaptive=False, lambda_=2.0)
    x = torch.randn(5, 3)
    out = model.forward(x)
    assert out.shape == (5, 1)


def test_admtsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        ADMTSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_admtsk_classifier_invalid_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        ADMTSKClassifierModel(_build_input_mfs(), n_classes=2, adaptive=False, lambda_=0.0)


def test_admtsk_regressor_invalid_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        ADMTSKRegressorModel(_build_input_mfs(), adaptive=False, lambda_=0.0)


def test_admtsk_classifier_accepts_custom_t_norm_fn() -> None:
    model = ADMTSKClassifierModel(
        _build_input_mfs(),
        n_classes=2,
        t_norm=DombiTNorm(lambda_=1.5),
    )
    x = torch.randn(4, 3)
    out = model.forward(x)
    assert out.shape == (4, 2)


def test_admtsk_regressor_accepts_custom_t_norm_fn() -> None:
    model = ADMTSKRegressorModel(
        _build_input_mfs(),
        t_norm=DombiTNorm(lambda_=1.5),
    )
    x = torch.randn(4, 3)
    out = model.forward(x)
    assert out.shape == (4, 1)
