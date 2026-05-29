from __future__ import annotations

import pytest
import torch
from torch import nn

from highfis.base import _iter_minibatch_indices
from highfis.layers import (
    GatedClassificationConsequentLayer,
    GatedClassificationZeroOrderConsequentLayer,
    GatedRegressionConsequentLayer,
    GatedRegressionZeroOrderConsequentLayer,
)
from highfis.memberships import GaussianMF
from highfis.models import (
    ADATSKClassifierModel,
    ADATSKRegressorModel,
    ADMTSKClassifierModel,
    ADMTSKRegressorModel,
    AYATSKClassifierModel,
    AYATSKRegressorModel,
    DGALETSKClassifierModel,
    DGALETSKRegressorModel,
    DombiTSKClassifierModel,
    DombiTSKRegressorModel,
    FSREADATSKClassifierModel,
    FSREADATSKRegressorModel,
    HDFISMinClassifierModel,
    HDFISMinRegressorModel,
    HDFISProdClassifierModel,
    HTSKClassifierModel,
    HTSKRegressorModel,
    LogTSKClassifierModel,
    LogTSKRegressorModel,
    MHTSKClassifierModel,
    MHTSKRegressorModel,
    TSKClassifierModel,
    TSKRegressorModel,
    build_rule_feature_mask,
)
from highfis.models._common import (
    _build_first_order_design_matrix,
    _threshold_from_zeta,
)
from highfis.t_norms import DombiTNorm


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_htsk_classifier_init_validates_arguments() -> None:
    with pytest.raises(ValueError, match="input_mfs must not be empty"):
        HTSKClassifierModel({}, n_classes=2)

    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        HTSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_htsk_classifier_forward_predict_shapes() -> None:
    model = HTSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_htsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = HTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(6, 3)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_ayatsk_classifier_forward_predict_shapes() -> None:
    model = AYATSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_ayatsk_regressor_forward_predict_shape() -> None:
    model = AYATSKRegressorModel(_build_input_mfs(), rule_base="coco")
    x = torch.randn(6, 3)

    output = model.forward(x)
    pred = model.predict(x)

    assert output.shape == (6, 1)
    assert pred.shape == (6,)


def test_ayatsk_classifier_init_validates_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        AYATSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_ayatsk_regressor_default_criterion() -> None:
    model = AYATSKRegressorModel(_build_input_mfs(), rule_base="coco")
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_adptsk_classifier_forward_predict_shapes() -> None:
    from highfis.models import ADPTSKClassifierModel

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
    from highfis.models import ADPTSKClassifierModel

    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        ADPTSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_adptsk_classifier_validates_kappa_xi() -> None:
    from highfis.models import ADPTSKClassifierModel

    with pytest.raises(ValueError, match="kappa must be > 0"):
        ADPTSKClassifierModel(_build_input_mfs(), n_classes=2, kappa=0.0)
    with pytest.raises(ValueError, match="xi must be > 0"):
        ADPTSKClassifierModel(_build_input_mfs(), n_classes=2, xi=-1.0)


def test_adptsk_regressor_validates_kappa_xi() -> None:
    from highfis.models import ADPTSKRegressorModel

    with pytest.raises(ValueError, match="kappa must be > 0"):
        ADPTSKRegressorModel(_build_input_mfs(), kappa=-5.0)
    with pytest.raises(ValueError, match="xi must be > 0"):
        ADPTSKRegressorModel(_build_input_mfs(), xi=0.0)


def test_adptsk_regressor_forward_predict_shape() -> None:
    from highfis.models import ADPTSKRegressorModel

    model = ADPTSKRegressorModel(_build_input_mfs(), rule_base="coco")
    x = torch.randn(6, 3)

    output = model.forward(x)
    pred = model.predict(x)

    assert output.shape == (6, 1)
    assert pred.shape == (6,)


def test_htsk_classifier_fit_returns_history() -> None:
    torch.manual_seed(1)
    model = HTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,), dtype=torch.long)

    history = model.fit(x, y, epochs=4, learning_rate=1e-2, batch_size=5, shuffle=True)

    assert set(history.keys()) == {"train", "ur", "val", "stopped_epoch"}
    assert len(history["train"]) == 4
    assert len(history["ur"]) == 4
    assert len(history["val"]) == 0
    assert history["stopped_epoch"] == 4


def test_htsk_classifier_fit_supports_custom_criterion() -> None:
    torch.manual_seed(1)
    model = HTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)

    history = model.fit(x, y, epochs=3, criterion=nn.MSELoss())

    assert len(history["train"]) == 3


def test_htsk_classifier_fit_validates_inputs() -> None:
    model = HTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,), dtype=torch.long)

    with pytest.raises(ValueError, match="expected x shape"):
        model.fit(torch.randn(10, 3), y, epochs=1)

    with pytest.raises(ValueError, match="expected y shape"):
        model.fit(x, y.unsqueeze(1), epochs=1)

    with pytest.raises(ValueError, match="ur_weight must be >= 0"):
        model.fit(x, y, epochs=1, ur_weight=-0.1)

    with pytest.raises(ValueError, match="ur_target must be in"):
        model.fit(x, y, epochs=1, ur_target=0.0)


def test_mhtsk_classifier_sparse_consequent_forward_shape() -> None:
    from highfis.memberships import ConstantMF

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
    from highfis.memberships import ConstantMF

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
    from highfis.memberships import ConstantMF

    input_mfs = {
        "x1": [GaussianMF(mean=-1.0, sigma=0.5), ConstantMF(1.0)],
        "x2": [GaussianMF(mean=-1.0, sigma=0.5), ConstantMF(1.0)],
    }
    rules = [(0, 1), (1, 0)]
    rule_feature_mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        MHTSKClassifierModel(input_mfs, rule_feature_mask, rules, n_classes=1)


def test_hdfismin_classifier_freezes_membership_parameters() -> None:
    model = HDFISMinClassifierModel(_build_input_mfs(), n_classes=2)
    assert all(not p.requires_grad for p in model.membership_layer.parameters())


def test_hdfisprod_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        HDFISProdClassifierModel(_build_input_mfs(), n_classes=1)


def test_hdfismin_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        HDFISMinClassifierModel(_build_input_mfs(), n_classes=1)


def test_hdfismin_regressor_freezes_membership_parameters() -> None:
    model = HDFISMinRegressorModel(_build_input_mfs(), rule_base="coco")
    assert all(not p.requires_grad for p in model.membership_layer.parameters())


def test_dombitsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = DombiTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_=2.0)
    x = torch.randn(6, 2)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_dombitsk_classifier_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        DombiTSKClassifierModel(_build_input_mfs(), n_classes=2, lambda_=0.0)


def test_dombitsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DombiTSKClassifierModel(_build_input_mfs(), n_classes=1, lambda_=1.0)


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
        paper_zero_consequent_init=False,
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
        paper_zero_consequent_init=False,
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


def test_dombitsk_classifier_default_t_norm_fn_branch() -> None:
    model = DombiTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_=2.0)
    x = torch.randn(4, 2)
    logits = model.forward(x)
    assert logits.shape == (4, 2)


def test_adatsk_classifier_forward_predict_shapes() -> None:
    model = ADATSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_adatsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = ADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(6, 2)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_adatsk_classifier_uses_ada_softmin_rule_layer() -> None:
    from highfis.layers import AdaSoftminRuleLayer

    model = ADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    assert isinstance(model.rule_layer, AdaSoftminRuleLayer)


def test_adatsk_regressor_forward_shape() -> None:
    model = ADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(5, 2)

    output = model.forward(x)

    assert output.shape == (5, 1)


def test_adatsk_regressor_fit_returns_history() -> None:
    torch.manual_seed(1)
    model = ADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)

    history = model.fit(x, y, epochs=3, learning_rate=1e-2, batch_size=5, shuffle=True)

    assert set(history.keys()) == {"train", "ur", "val", "stopped_epoch"}
    assert len(history["train"]) == 3
    assert len(history["ur"]) == 3
    assert history["stopped_epoch"] == 3


def test_adatsk_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        ADATSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_fsre_adatsk_classifier_helpers() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(12, 2)
    y = torch.randint(0, 2, (12,), dtype=torch.long)

    history_fs = model.fit_fs(x, y, epochs=2, batch_size=6)
    assert history_fs["stopped_epoch"] == 2

    history_re = model.fit_re(x, y, epochs=2, batch_size=6)
    assert history_re["stopped_epoch"] == 2

    history_ft = model.fit_finetune(x, y, epochs=2, batch_size=6)
    assert history_ft["stopped_epoch"] == 2


def test_fsre_adatsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=1)


def test_fsre_adatsk_regressor_helpers() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(12, 2)
    y = torch.randn(12)

    history_fs = model.fit_fs(x, y, epochs=2, batch_size=6)
    assert history_fs["stopped_epoch"] == 2

    history_re = model.fit_re(x, y, epochs=2, batch_size=6)
    assert history_re["stopped_epoch"] == 2

    history_ft = model.fit_finetune(x, y, epochs=2, batch_size=6)
    assert history_ft["stopped_epoch"] == 2


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

    # Non-LSE path leaves model in zero-order mode; conversion happens in fit_finetune.
    assert isinstance(model.consequent_layer, GatedRegressionZeroOrderConsequentLayer)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_regressor_convert_to_first_order_idempotent() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)


def test_dombi_tsk_regressor_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        DombiTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_=0.0)


def test_dombi_tsk_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DombiTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=1, lambda_=1.0)


def test_dombi_tsk_classifier_default_t_norm_fn_branch() -> None:
    model = DombiTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_=1.5)
    x = torch.randn(4, 2)
    logits = model.forward(x)
    assert logits.shape == (4, 2)


def test_dombi_tsk_classifier_explicit_t_norm_fn_branch() -> None:
    model = DombiTSKClassifierModel(
        _build_input_mfs(n_inputs=2, n_mfs=2),
        n_classes=2,
        lambda_=1.5,
        t_norm=lambda terms, dim=-1: terms.prod(dim=dim),
    )
    x = torch.randn(4, 2)
    logits = model.forward(x)
    assert logits.shape == (4, 2)


def test_dombi_tsk_regressor_default_t_norm_fn_branch() -> None:
    model = DombiTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_=1.0)
    x = torch.randn(4, 2)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_dombi_tsk_regressor_explicit_t_norm_fn_branch() -> None:
    model = DombiTSKRegressorModel(
        _build_input_mfs(n_inputs=2, n_mfs=2),
        lambda_=1.0,
        t_norm=lambda terms, dim=-1: terms.prod(dim=dim),
    )
    x = torch.randn(4, 2)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_threshold_from_zeta_invalid_range() -> None:
    gate_values = torch.tensor([0.2, 0.8])
    with pytest.raises(ValueError, match=r"zeta must be in \[0, 1\]"):
        _threshold_from_zeta(gate_values, -0.1)
    with pytest.raises(ValueError, match=r"zeta must be in \[0, 1\]"):
        _threshold_from_zeta(gate_values, 1.1)


def test_tsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        TSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_tsk_regressor_forward_shape() -> None:
    model = TSKRegressorModel(_build_input_mfs())
    x = torch.randn(4, 3)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_log_tsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        LogTSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_log_tsk_classifier_forward_shapes() -> None:
    model = LogTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(4, 3)
    logits = model.forward(x)
    assert logits.shape == (4, 2)
    assert torch.allclose(torch.softmax(logits, dim=1).sum(dim=1), torch.ones(4), atol=1e-6)


def test_log_tsk_regressor_forward_shape() -> None:
    model = LogTSKRegressorModel(_build_input_mfs())
    x = torch.randn(4, 3)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_tsk_classifier_forward_shapes() -> None:
    model = TSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(4, 3)
    logits = model.forward(x)
    assert logits.shape == (4, 2)
    assert torch.allclose(torch.softmax(logits, dim=1).sum(dim=1), torch.ones(4), atol=1e-6)


def test_tsk_classifier_default_criterion() -> None:
    model = TSKClassifierModel(_build_input_mfs(), n_classes=2)
    assert isinstance(model._default_criterion(), nn.CrossEntropyLoss)


def test_tsk_regressor_default_criterion() -> None:
    model = TSKRegressorModel(_build_input_mfs())
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_dombi_tsk_classifier_default_criterion() -> None:
    model = DombiTSKClassifierModel(_build_input_mfs(), n_classes=2, lambda_=1.0)
    assert isinstance(model._default_criterion(), nn.CrossEntropyLoss)


def test_dombi_tsk_regressor_default_criterion() -> None:
    model = DombiTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_=1.0)
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_log_tsk_classifier_default_criterion() -> None:
    model = LogTSKClassifierModel(_build_input_mfs(), n_classes=2)
    assert isinstance(model._default_criterion(), nn.CrossEntropyLoss)


def test_log_tsk_regressor_default_criterion() -> None:
    model = LogTSKRegressorModel(_build_input_mfs())
    assert isinstance(model._default_criterion(), nn.MSELoss)


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


def test_dg_aletsk_classifier_default_criterion() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_dg_aletsk_regressor_default_criterion() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    assert isinstance(model._default_criterion(), nn.MSELoss)


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


def test_dg_aletsk_classifier_fit_dg_phase_and_finetune() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)

    history = model.fit_dg_phase(x, y, epochs=2, batch_size=8)
    assert history["stopped_epoch"] == 2

    history = model.fit_finetune(x, y, epochs=2, batch_size=8)
    assert history["stopped_epoch"] == 2


def test_dg_aletsk_classifier_search_thresholds_no_candidates_raises() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,), dtype=torch.long)

    with pytest.raises(RuntimeError, match="threshold search did not yield a valid candidate"):
        model.search_thresholds(x, y, zeta_lambda=[], zeta_theta=[], inplace=False)


def test_dg_aletsk_regressor_fit_dg_phase_and_finetune() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(16, 2)
    y = torch.randn(16)

    history = model.fit_dg_phase(x, y, epochs=2, batch_size=8)
    assert history["stopped_epoch"] == 2

    history = model.fit_finetune(x, y, epochs=2, batch_size=8)
    assert history["stopped_epoch"] == 2


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


def test_adatsk_classifier_consequent_batch_norm() -> None:
    model = ADATSKClassifierModel(_build_input_mfs(), n_classes=2, consequent_batch_norm=True)
    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,), dtype=torch.long)

    history = model.fit(x, y, epochs=2, learning_rate=1e-2, batch_size=4)
    assert history["stopped_epoch"] == 2
    assert model.predict(x).shape == (8,)


def test_adatsk_classifier_custom_rule_base_and_rules() -> None:
    input_mfs = _build_input_mfs(n_inputs=2, n_mfs=2)
    custom_rules = [(0, 0), (1, 1)]
    model = ADATSKClassifierModel(
        input_mfs,
        n_classes=2,
        rule_base="custom",
        rules=custom_rules,
    )
    assert model.n_rules == 2
    x = torch.randn(4, 2)
    logits = model.forward(x)
    assert logits.shape == (4, 2)
    assert torch.allclose(model.forward_antecedents(x).sum(dim=1), torch.ones(4), atol=1e-6)


def test_adatsk_regressor_consequent_batch_norm() -> None:
    model = ADATSKRegressorModel(
        _build_input_mfs(n_inputs=2, n_mfs=2),
        consequent_batch_norm=True,
    )
    x = torch.randn(8, 2)
    y = torch.randn(8)
    history = model.fit(x, y, epochs=2, learning_rate=1e-2, batch_size=4)
    assert history["stopped_epoch"] == 2
    assert model.predict(x).shape == (8,)


def test_htsk_classifier_fit_history_keys_without_val() -> None:
    model = HTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,), dtype=torch.long)

    history = model.fit(x, y, epochs=3)

    assert set(history.keys()) == {"train", "ur", "val", "stopped_epoch"}
    assert len(history["val"]) == 0
    assert history["stopped_epoch"] == 3


def test_htsk_classifier_early_stopping_with_val_data() -> None:
    torch.manual_seed(42)
    model = HTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(30, 2)
    y = torch.randint(0, 2, (30,), dtype=torch.long)
    x_val = torch.randn(10, 2)
    y_val = torch.randint(0, 2, (10,), dtype=torch.long)

    history = model.fit(
        x,
        y,
        epochs=500,
        x_val=x_val,
        y_val=y_val,
        patience=5,
        learning_rate=1e-2,
    )

    assert len(history["val"]) == len(history["train"])
    assert len(history["val"]) > 0
    assert len(history["val_acc"]) == len(history["train"])
    assert "stopped_epoch" in history
    # Early stopping should fire well before 500 epochs
    assert history["stopped_epoch"] < 500


def test_htsk_classifier_fit_validates_val_inputs() -> None:
    model = HTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,), dtype=torch.long)

    with pytest.raises(ValueError, match="expected x_val shape"):
        model.fit(x, y, epochs=1, x_val=torch.randn(5, 3), y_val=torch.randint(0, 2, (5,)))

    with pytest.raises(ValueError, match="expected y_val shape"):
        model.fit(x, y, epochs=1, x_val=torch.randn(5, 2), y_val=torch.randint(0, 2, (5, 1)))


# ---------------------------------------------------------------------------
# _iter_minibatch_indices
# ---------------------------------------------------------------------------


def test_iter_minibatch_indices_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        _iter_minibatch_indices(100, batch_size=0, shuffle=False)


def test_build_first_order_design_matrix_validates_input_shapes() -> None:
    norm_w = torch.rand(2, 3)
    x = torch.randn(2, 2)

    with pytest.raises(ValueError, match="feature_gates must have shape"):
        _build_first_order_design_matrix(norm_w, x, torch.randn(1, 2), torch.randn(3))

    with pytest.raises(ValueError, match="rule_gates must have shape"):
        _build_first_order_design_matrix(norm_w, x, torch.randn(3, 2), torch.randn(1))


def test_dombitsk_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DombiTSKClassifierModel(_build_input_mfs(), n_classes=1, lambda_=1.0)


def test_dg_aletsk_classifier_fit_first_order_consequents_requires_conversion() -> None:
    model = DGALETSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,), dtype=torch.long)

    with pytest.raises(ValueError, match=r"convert_to_first_order\(\) must be called before LSE consequent fitting"):
        model._fit_first_order_consequents_lse(x, y)


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


def test_dg_aletsk_regressor_fit_first_order_consequents_requires_conversion() -> None:
    model = DGALETSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(8, 2)
    y = torch.randn(8)

    with pytest.raises(ValueError, match=r"convert_to_first_order\(\) must be called before LSE consequent fitting"):
        model._fit_first_order_consequents_lse(x, y)


# ---------------------------------------------------------------------------
# consequent_batch_norm
# ---------------------------------------------------------------------------


def test_htsk_classifier_consequent_batch_norm() -> None:
    """consequent_batch_norm=True covers BN in forward (line 169) and fit optimizer (line 174)."""
    torch.manual_seed(1)
    model = HTSKClassifierModel(
        _build_input_mfs(n_inputs=2, n_mfs=2),
        n_classes=2,
        consequent_batch_norm=True,
    )
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,), dtype=torch.long)

    history = model.fit(x, y, epochs=2, batch_size=10)
    assert len(history["train"]) == 2

    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (20, 2)


# ---------------------------------------------------------------------------
# MSELoss + validation (lines 228-230)
# ---------------------------------------------------------------------------


def test_htsk_classifier_fit_mse_with_validation() -> None:
    """MSELoss criterion + validation data covers the MSELoss path in val loop."""
    torch.manual_seed(0)
    model = HTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,), dtype=torch.long)
    x_val = torch.randn(8, 2)
    y_val = torch.randint(0, 2, (8,), dtype=torch.long)

    history = model.fit(
        x,
        y,
        epochs=2,
        criterion=nn.MSELoss(),
        x_val=x_val,
        y_val=y_val,
        patience=10,
    )
    assert len(history["val"]) == 2


# ---------------------------------------------------------------------------
# verbose logging paths (lines 246, 257, 261)
# ---------------------------------------------------------------------------


def test_htsk_classifier_fit_verbose_with_early_stopping() -> None:
    """verbose=2 + early stopping exercises per-epoch logging with validation."""
    torch.manual_seed(42)
    model = HTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(30, 2)
    y = torch.randint(0, 2, (30,), dtype=torch.long)
    x_val = torch.randn(10, 2)
    y_val = torch.randint(0, 2, (10,), dtype=torch.long)

    history = model.fit(
        x,
        y,
        epochs=500,
        x_val=x_val,
        y_val=y_val,
        patience=5,
        learning_rate=1e-2,
        verbose=2,
    )
    assert history["stopped_epoch"] < 500


def test_htsk_classifier_fit_verbose_no_validation() -> None:
    """verbose=2 without validation exercises the no-val per-epoch logging path."""
    torch.manual_seed(0)
    model = HTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,), dtype=torch.long)

    history = model.fit(x, y, epochs=10, verbose=2)
    assert len(history["train"]) == 10


# ===========================================================================
# HTSKRegressorModel
# ===========================================================================


def test_htsk_regressor_init_validates_arguments() -> None:
    with pytest.raises(ValueError, match="input_mfs must not be empty"):
        HTSKRegressorModel({})


def test_htsk_regressor_forward_predict_shapes() -> None:
    model = HTSKRegressorModel(_build_input_mfs())
    x = torch.randn(8, 3)

    out = model.forward(x)
    pred = model.predict(x)

    assert out.shape == (8, 1)
    assert pred.shape == (8,)


def test_htsk_regressor_forward_antecedents_row_sum_one() -> None:
    model = HTSKRegressorModel(_build_input_mfs())
    x = torch.randn(6, 3)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_htsk_regressor_fit_returns_history() -> None:
    torch.manual_seed(1)
    model = HTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)

    history = model.fit(x, y, epochs=4, learning_rate=1e-2, batch_size=5, shuffle=True)

    assert set(history.keys()) == {"train", "ur", "val", "stopped_epoch"}
    assert len(history["train"]) == 4
    assert len(history["ur"]) == 4
    assert len(history["val"]) == 0
    assert history["stopped_epoch"] == 4


def test_htsk_regressor_fit_loss_decreases() -> None:
    torch.manual_seed(42)
    model = HTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(40, 2)
    y = x[:, 0] + 0.5 * x[:, 1]

    history = model.fit(x, y, epochs=50, learning_rate=1e-2)

    assert history["train"][-1] < history["train"][0]


def test_htsk_regressor_fit_validates_inputs() -> None:
    model = HTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(10, 2)
    y = torch.randn(10)

    with pytest.raises(ValueError, match="expected x shape"):
        model.fit(torch.randn(10, 3), y, epochs=1)

    with pytest.raises(ValueError, match="expected y shape"):
        model.fit(x, y.unsqueeze(1), epochs=1)

    with pytest.raises(ValueError, match="ur_weight must be >= 0"):
        model.fit(x, y, epochs=1, ur_weight=-0.1)

    with pytest.raises(ValueError, match="ur_target must be in"):
        model.fit(x, y, epochs=1, ur_target=0.0)


def test_htsk_regressor_early_stopping_with_val_data() -> None:
    torch.manual_seed(42)
    model = HTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(30, 2)
    y = x[:, 0] + 0.5 * x[:, 1]
    x_val = torch.randn(10, 2)
    y_val = x_val[:, 0] + 0.5 * x_val[:, 1]

    history = model.fit(
        x,
        y,
        epochs=2000,
        x_val=x_val,
        y_val=y_val,
        patience=15,
        learning_rate=5e-2,
    )

    assert len(history["val"]) == len(history["train"])
    assert len(history["val"]) > 0
    assert "stopped_epoch" in history
    assert history["stopped_epoch"] < 2000


def test_htsk_regressor_fit_validates_val_inputs() -> None:
    model = HTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(10, 2)
    y = torch.randn(10)

    with pytest.raises(ValueError, match="expected x_val shape"):
        model.fit(x, y, epochs=1, x_val=torch.randn(5, 3), y_val=torch.randn(5))

    with pytest.raises(ValueError, match="expected y_val shape"):
        model.fit(x, y, epochs=1, x_val=torch.randn(5, 2), y_val=torch.randn(5, 1))


def test_htsk_regressor_consequent_batch_norm() -> None:
    torch.manual_seed(1)
    model = HTSKRegressorModel(
        _build_input_mfs(n_inputs=2, n_mfs=2),
        consequent_batch_norm=True,
    )
    x = torch.randn(20, 2)
    y = torch.randn(20)

    history = model.fit(x, y, epochs=2, batch_size=10)
    assert len(history["train"]) == 2

    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (20, 1)


def test_htsk_regressor_fit_verbose_with_early_stopping() -> None:
    torch.manual_seed(42)
    model = HTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(30, 2)
    y = x[:, 0] + 0.5 * x[:, 1]
    x_val = torch.randn(10, 2)
    y_val = x_val[:, 0] + 0.5 * x_val[:, 1]

    history = model.fit(
        x,
        y,
        epochs=2000,
        x_val=x_val,
        y_val=y_val,
        patience=15,
        learning_rate=5e-2,
        verbose=2,
    )
    assert history["stopped_epoch"] < 2000


def test_htsk_regressor_fit_verbose_no_validation() -> None:
    torch.manual_seed(0)
    model = HTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)

    history = model.fit(x, y, epochs=10, verbose=2)
    assert len(history["train"]) == 10


def test_htsk_regressor_single_sample() -> None:
    torch.manual_seed(0)
    model = HTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(1, 2)
    y = torch.tensor([1.0])

    history = model.fit(x, y, epochs=3)
    assert len(history["train"]) == 3

    pred = model.predict(x)
    assert pred.shape == (1,)


def test_htsk_regressor_constant_targets() -> None:
    torch.manual_seed(0)
    model = HTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.full((20,), 3.14)

    _history = model.fit(x, y, epochs=200, learning_rate=5e-2)
    pred = model.predict(x)

    # Should converge close to the constant target
    assert float(torch.abs(pred.mean() - 3.14)) < 1.0
