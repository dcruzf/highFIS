from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn

from highfis import ADATSKClassifier
from highfis.estimators import InputConfig
from highfis.estimators._adaptive import _set_sigma_to_one_and_freeze, _wrap_adatsk_gaussian_input_mfs
from highfis.layers import AdaSoftminRuleLayer
from highfis.memberships import ADATSKGaussianMF, CompositeGaussianMF, GaussianMF
from highfis.models import ADATSKClassifierModel, ADATSKRegressorModel


def _build_adatsk_input_mfs(n_inputs: int = 3, n_rules: int = 2) -> dict[str, list[CompositeGaussianMF]]:
    return {
        f"x{i + 1}": [CompositeGaussianMF(mean=float(j), sigma=1.0, eps=1e-4) for j in range(n_rules)]
        for i in range(n_inputs)
    }


def test_composite_gaussian_mf_lower_bound() -> None:
    mf = CompositeGaussianMF(mean=0.0, sigma=1.0, eps=0.05)
    x = torch.tensor([-5.0, 0.0, 5.0])

    values = mf(x)

    assert torch.all(values >= 0.05)
    assert torch.all(values <= 1.0)


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


def test_adatsk_classifier_defaults_follow_paper_profile() -> None:
    clf = ADATSKClassifier()

    assert clf.n_mfs == 3
    assert clf.mf_init == "grid"
    assert clf.rule_base == "coco"
    assert clf.batch_size is None
    assert clf.shuffle is False
    assert clf.patience is None
    assert clf.restore_best is False
    assert clf.weight_decay == 0.0


def test_adatsk_gaussian_mf_matches_paper_eq3_when_sigma_one() -> None:
    mf = ADATSKGaussianMF(mean=0.0, sigma=1.0)
    x = torch.tensor([-1.0, 0.0, 2.0])

    values = mf(x)
    expected = torch.exp(-x.square())

    assert torch.allclose(values, expected, atol=1e-6)


def test_adatsk_classifier_pre_hook_sets_sigma_one_and_freezes_sigma() -> None:
    x = np.random.default_rng(0).normal(size=(12, 3)).astype(np.float64)
    y = np.random.default_rng(1).integers(0, 2, size=(12,), dtype=np.int64)
    clf = ADATSKClassifier(epochs=1, high_dim_threshold=10_000)

    clf.fit(x, y)

    for mf_list in clf.model_.membership_layer.input_mfs.values():
        for module in cast(nn.ModuleList, mf_list):
            mf = cast(GaussianMF, module)
            assert abs(float(mf.sigma.detach().item()) - 1.0) < 1e-3
            assert mf.raw_sigma.requires_grad is False


def test_adatsk_classifier_high_dim_freezes_antecedents() -> None:
    x = np.random.default_rng(2).normal(size=(10, 4)).astype(np.float64)
    y = np.random.default_rng(3).integers(0, 2, size=(10,), dtype=np.int64)
    clf = ADATSKClassifier(epochs=1, high_dim_threshold=4)

    clf.fit(x, y)

    assert all(not p.requires_grad for p in clf.model_.membership_layer.parameters())


def test_adatsk_classifier_grid_init_uses_no_margin_centers() -> None:
    x = np.array([[-1.0], [1.0]], dtype=np.float64)
    y = np.array([0, 1], dtype=np.int64)
    clf = ADATSKClassifier(n_mfs=3, mf_init="grid", epochs=1, high_dim_threshold=1)

    clf.fit(x, y)

    mf_list = cast(nn.ModuleList, clf.model_.membership_layer.input_mfs["x1"])
    centers = torch.tensor([cast(ADATSKGaussianMF, mf).mean.detach().item() for mf in mf_list], dtype=torch.float32)
    expected = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(centers, expected, atol=1e-6)


def test_adatsk_classifier_consequents_are_zero_initialized() -> None:
    model = ADATSKClassifierModel(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)

    weight = cast(torch.Tensor, model.consequent_layer.weight).detach()
    bias = cast(torch.Tensor, model.consequent_layer.bias).detach()

    assert torch.allclose(weight, torch.zeros_like(weight))
    assert torch.allclose(bias, torch.zeros_like(bias))


def test_set_sigma_to_one_and_freeze_ignores_non_gaussian_mf() -> None:
    mf = CompositeGaussianMF(mean=0.0, sigma=0.7, eps=1e-4)
    before = float(mf.sigma.detach().item())
    assert mf.raw_sigma.requires_grad is True

    _set_sigma_to_one_and_freeze(mf)

    after = float(mf.sigma.detach().item())
    assert abs(after - before) < 1e-6
    assert mf.raw_sigma.requires_grad is True


def test_wrap_adatsk_gaussian_input_mfs_preserves_non_gaussian_modules() -> None:
    cg = CompositeGaussianMF(mean=0.0, sigma=0.9, eps=1e-4)
    g = GaussianMF(mean=1.0, sigma=1.1, eps=1e-4)
    wrapped = _wrap_adatsk_gaussian_input_mfs({"x1": [g, cg]})

    assert isinstance(wrapped["x1"][0], ADATSKGaussianMF)
    assert wrapped["x1"][1] is cg


def test_adatsk_classifier_resolve_input_configs_keeps_user_configs() -> None:
    configs = [InputConfig(name="x1", n_mfs=3, overlap=0.5, margin=0.2)]
    clf = ADATSKClassifier(input_configs=configs)
    x = np.array([[0.0], [1.0]], dtype=np.float64)

    resolved = clf._resolve_input_configs(x)

    assert resolved[0].margin == 0.2


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
