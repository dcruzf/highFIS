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
    AdaTSKClassifier,
    AdaTSKRegressor,
    ADMTSKClassifier,
    ADMTSKRegressor,
    AYATSKClassifier,
    AYATSKRegressor,
    DGALETSKClassifier,
    DGALETSKRegressor,
    DombiTSKClassifier,
    DombiTSKRegressor,
    FSREAdaTSKClassifier,
    FSREAdaTSKRegressor,
    HDFISMinClassifier,
    HDFISMinRegressor,
    HDFISProdClassifier,
    HTSKClassifier,
    HTSKRegressor,
    LogTSKClassifier,
    LogTSKRegressor,
    TSKClassifier,
    TSKRegressor,
    _build_first_order_design_matrix,
    _threshold_from_zeta,
)
from highfis.t_norms import DombiTNorm


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_htsk_classifier_init_validates_arguments() -> None:
    with pytest.raises(ValueError, match="input_mfs must not be empty"):
        HTSKClassifier({}, n_classes=2)

    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        HTSKClassifier(_build_input_mfs(), n_classes=1)


def test_htsk_classifier_forward_predict_shapes() -> None:
    model = HTSKClassifier(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_htsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = HTSKClassifier(_build_input_mfs(), n_classes=2)
    x = torch.randn(6, 3)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_ayatsk_classifier_forward_predict_shapes() -> None:
    model = AYATSKClassifier(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_ayatsk_regressor_forward_predict_shape() -> None:
    model = AYATSKRegressor(_build_input_mfs(), rule_base="coco")
    x = torch.randn(6, 3)

    output = model.forward(x)
    pred = model.predict(x)

    assert output.shape == (6, 1)
    assert pred.shape == (6,)


def test_ayatsk_classifier_init_validates_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        AYATSKClassifier(_build_input_mfs(), n_classes=1)


def test_ayatsk_regressor_default_criterion() -> None:
    model = AYATSKRegressor(_build_input_mfs(), rule_base="coco")
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_htsk_classifier_fit_returns_history() -> None:
    torch.manual_seed(1)
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
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
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)

    history = model.fit(x, y, epochs=3, criterion=nn.MSELoss())

    assert len(history["train"]) == 3


def test_htsk_classifier_fit_validates_inputs() -> None:
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
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


def test_hdfismin_classifier_freezes_membership_parameters() -> None:
    model = HDFISMinClassifier(_build_input_mfs(), n_classes=2)
    assert all(not p.requires_grad for p in model.membership_layer.parameters())


def test_hdfisprod_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        HDFISProdClassifier(_build_input_mfs(), n_classes=1)


def test_hdfismin_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        HDFISMinClassifier(_build_input_mfs(), n_classes=1)


def test_hdfismin_regressor_freezes_membership_parameters() -> None:
    model = HDFISMinRegressor(_build_input_mfs(), rule_base="coco")
    assert all(not p.requires_grad for p in model.membership_layer.parameters())


def test_dombitsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = DombiTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_=2.0)
    x = torch.randn(6, 2)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_dombitsk_classifier_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        DombiTSKClassifier(_build_input_mfs(), n_classes=2, lambda_=0.0)


def test_dombitsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DombiTSKClassifier(_build_input_mfs(), n_classes=1, lambda_=1.0)


def test_admtsk_classifier_forward_predict_shapes() -> None:
    model = ADMTSKClassifier(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_admtsk_regressor_forward_shape() -> None:
    model = ADMTSKRegressor(_build_input_mfs(), rule_base="coco")
    x = torch.randn(5, 3)

    output = model.forward(x)
    pred = model.predict(x)

    assert output.shape == (5, 1)
    assert pred.shape == (5,)


def test_admtsk_classifier_fixed_lambda_branch() -> None:
    model = ADMTSKClassifier(_build_input_mfs(), n_classes=2, adaptive=False, lambda_=2.0)
    x = torch.randn(6, 3)
    out = model.forward(x)
    assert out.shape == (6, 2)


def test_admtsk_regressor_fixed_lambda_branch() -> None:
    model = ADMTSKRegressor(_build_input_mfs(), adaptive=False, lambda_=2.0)
    x = torch.randn(5, 3)
    out = model.forward(x)
    assert out.shape == (5, 1)


def test_admtsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        ADMTSKClassifier(_build_input_mfs(), n_classes=1)


def test_admtsk_classifier_invalid_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        ADMTSKClassifier(_build_input_mfs(), n_classes=2, adaptive=False, lambda_=0.0)


def test_admtsk_regressor_invalid_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        ADMTSKRegressor(_build_input_mfs(), adaptive=False, lambda_=0.0)


def test_admtsk_classifier_accepts_custom_t_norm_fn() -> None:
    model = ADMTSKClassifier(
        _build_input_mfs(),
        n_classes=2,
        t_norm_fn=DombiTNorm(lambda_=1.5),
    )
    x = torch.randn(4, 3)
    out = model.forward(x)
    assert out.shape == (4, 2)


def test_admtsk_regressor_accepts_custom_t_norm_fn() -> None:
    model = ADMTSKRegressor(
        _build_input_mfs(),
        t_norm_fn=DombiTNorm(lambda_=1.5),
    )
    x = torch.randn(4, 3)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_dombitsk_classifier_default_t_norm_fn_branch() -> None:
    model = DombiTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_=2.0)
    x = torch.randn(4, 2)
    logits = model.forward(x)
    assert logits.shape == (4, 2)


def test_adatsk_classifier_forward_predict_shapes() -> None:
    model = AdaTSKClassifier(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_adatsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = AdaTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(6, 2)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_adatsk_classifier_uses_ada_softmin_rule_layer() -> None:
    from highfis.layers import AdaSoftminRuleLayer

    model = AdaTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    assert isinstance(model.rule_layer, AdaSoftminRuleLayer)


def test_adatsk_regressor_forward_shape() -> None:
    model = AdaTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(5, 2)

    output = model.forward(x)

    assert output.shape == (5, 1)


def test_adatsk_regressor_fit_returns_history() -> None:
    torch.manual_seed(1)
    model = AdaTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)

    history = model.fit(x, y, epochs=3, learning_rate=1e-2, batch_size=5, shuffle=True)

    assert set(history.keys()) == {"train", "ur", "val", "stopped_epoch"}
    assert len(history["train"]) == 3
    assert len(history["ur"]) == 3
    assert history["stopped_epoch"] == 3


def test_adatsk_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        AdaTSKClassifier(_build_input_mfs(), n_classes=1)


def test_fsre_adatsk_classifier_helpers() -> None:
    model = FSREAdaTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
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
        FSREAdaTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=1)


def test_fsre_adatsk_regressor_helpers() -> None:
    model = FSREAdaTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(12, 2)
    y = torch.randn(12)

    history_fs = model.fit_fs(x, y, epochs=2, batch_size=6)
    assert history_fs["stopped_epoch"] == 2

    history_re = model.fit_re(x, y, epochs=2, batch_size=6)
    assert history_re["stopped_epoch"] == 2

    history_ft = model.fit_finetune(x, y, epochs=2, batch_size=6)
    assert history_ft["stopped_epoch"] == 2


def test_dg_aletsk_classifier_invalid_lambda_init() -> None:
    with pytest.raises(ValueError, match="lambda_init must be > 0"):
        DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=0.0)


def test_dg_aletsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=1, lambda_init=1.0)


def test_dg_aletsk_classifier_thresholds_and_convert() -> None:
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
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
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=False, verbose=True)
    after = model.state_dict()

    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_dg_aletsk_classifier_search_thresholds_inplace_true_converts_zero_order_self() -> None:
    torch.manual_seed(0)
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)

    assert isinstance(model.consequent_layer, GatedClassificationZeroOrderConsequentLayer)
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=True, verbose=False)

    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_classifier_convert_to_first_order_idempotent() -> None:
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
    model.convert_to_first_order()
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)


def test_dg_aletsk_regressor_invalid_lambda_init() -> None:
    with pytest.raises(ValueError, match="lambda_init must be > 0"):
        DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=0.0)


def test_dg_aletsk_regressor_thresholds_apply_and_search() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]

    with pytest.raises(ValueError, match="thresholds must be finite"):
        model.apply_thresholds(float("inf"), 0.0)

    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=False)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_regressor_search_thresholds_inplace_true_and_verbose() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    model.convert_to_first_order()
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]

    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=False, inplace=True, verbose=True)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_regressor_search_thresholds_inplace_true_converts_zero_order_self() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]

    assert isinstance(model.consequent_layer, GatedRegressionZeroOrderConsequentLayer)
    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=False, inplace=True, verbose=False)

    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_regressor_convert_to_first_order_idempotent() -> None:
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    model.convert_to_first_order()
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)


def test_dombi_tsk_regressor_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        DombiTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_=0.0)


def test_dombi_tsk_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DombiTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=1, lambda_=1.0)


def test_dombi_tsk_classifier_default_t_norm_fn_branch() -> None:
    model = DombiTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_=1.5)
    x = torch.randn(4, 2)
    logits = model.forward(x)
    assert logits.shape == (4, 2)


def test_dombi_tsk_classifier_explicit_t_norm_fn_branch() -> None:
    model = DombiTSKClassifier(
        _build_input_mfs(n_inputs=2, n_mfs=2),
        n_classes=2,
        lambda_=1.5,
        t_norm_fn=lambda terms, dim=-1: terms.prod(dim=dim),
    )
    x = torch.randn(4, 2)
    logits = model.forward(x)
    assert logits.shape == (4, 2)


def test_dombi_tsk_regressor_default_t_norm_fn_branch() -> None:
    model = DombiTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_=1.0)
    x = torch.randn(4, 2)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_dombi_tsk_regressor_explicit_t_norm_fn_branch() -> None:
    model = DombiTSKRegressor(
        _build_input_mfs(n_inputs=2, n_mfs=2),
        lambda_=1.0,
        t_norm_fn=lambda terms, dim=-1: terms.prod(dim=dim),
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
        TSKClassifier(_build_input_mfs(), n_classes=1)


def test_tsk_regressor_forward_shape() -> None:
    model = TSKRegressor(_build_input_mfs())
    x = torch.randn(4, 3)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_log_tsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        LogTSKClassifier(_build_input_mfs(), n_classes=1)


def test_log_tsk_classifier_forward_shapes() -> None:
    model = LogTSKClassifier(_build_input_mfs(), n_classes=2)
    x = torch.randn(4, 3)
    logits = model.forward(x)
    assert logits.shape == (4, 2)
    assert torch.allclose(torch.softmax(logits, dim=1).sum(dim=1), torch.ones(4), atol=1e-6)


def test_log_tsk_regressor_forward_shape() -> None:
    model = LogTSKRegressor(_build_input_mfs())
    x = torch.randn(4, 3)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_tsk_classifier_forward_shapes() -> None:
    model = TSKClassifier(_build_input_mfs(), n_classes=2)
    x = torch.randn(4, 3)
    logits = model.forward(x)
    assert logits.shape == (4, 2)
    assert torch.allclose(torch.softmax(logits, dim=1).sum(dim=1), torch.ones(4), atol=1e-6)


def test_tsk_classifier_default_criterion() -> None:
    model = TSKClassifier(_build_input_mfs(), n_classes=2)
    assert isinstance(model._default_criterion(), nn.CrossEntropyLoss)


def test_tsk_regressor_default_criterion() -> None:
    model = TSKRegressor(_build_input_mfs())
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_dombi_tsk_classifier_default_criterion() -> None:
    model = DombiTSKClassifier(_build_input_mfs(), n_classes=2, lambda_=1.0)
    assert isinstance(model._default_criterion(), nn.CrossEntropyLoss)


def test_dombi_tsk_regressor_default_criterion() -> None:
    model = DombiTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_=1.0)
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_log_tsk_classifier_default_criterion() -> None:
    model = LogTSKClassifier(_build_input_mfs(), n_classes=2)
    assert isinstance(model._default_criterion(), nn.CrossEntropyLoss)


def test_log_tsk_regressor_default_criterion() -> None:
    model = LogTSKRegressor(_build_input_mfs())
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_dg_aletsk_classifier_convert_to_first_order_preserves_theta() -> None:
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
    assert isinstance(model.consequent_layer, GatedClassificationZeroOrderConsequentLayer)
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)


def test_dg_aletsk_regressor_convert_to_first_order_preserves_theta() -> None:
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    assert isinstance(model.consequent_layer, GatedRegressionZeroOrderConsequentLayer)
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)


def test_dg_aletsk_classifier_default_criterion() -> None:
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
    assert isinstance(model._default_criterion(), nn.CrossEntropyLoss)


def test_dg_aletsk_regressor_default_criterion() -> None:
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_dg_aletsk_regressor_search_thresholds_verbose() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]

    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=False, verbose=True)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_regressor_search_thresholds_use_lse_false() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]

    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=False, inplace=False, verbose=False)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_classifier_fit_dg_phase_and_finetune() -> None:
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)

    history = model.fit_dg_phase(x, y, epochs=2, batch_size=8)
    assert history["stopped_epoch"] == 2

    history = model.fit_finetune(x, y, epochs=2, batch_size=8)
    assert history["stopped_epoch"] == 2


def test_dg_aletsk_classifier_search_thresholds_no_candidates_raises() -> None:
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
    x = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,), dtype=torch.long)

    with pytest.raises(RuntimeError, match="threshold search did not yield a valid candidate"):
        model.search_thresholds(x, y, zeta_lambda=[], zeta_theta=[], inplace=False)


def test_dg_aletsk_regressor_fit_dg_phase_and_finetune() -> None:
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    x = torch.randn(16, 2)
    y = torch.randn(16)

    history = model.fit_dg_phase(x, y, epochs=2, batch_size=8)
    assert history["stopped_epoch"] == 2

    history = model.fit_finetune(x, y, epochs=2, batch_size=8)
    assert history["stopped_epoch"] == 2


def test_dg_aletsk_regressor_search_thresholds_no_candidates_raises() -> None:
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    x = torch.randn(8, 2)
    y = torch.randn(8)

    with pytest.raises(RuntimeError, match="threshold search did not yield a valid candidate"):
        model.search_thresholds(x, y, zeta_lambda=[], zeta_theta=[], inplace=False)


def test_dg_aletsk_classifier_convert_to_first_order_preserves_theta_values() -> None:
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
    theta_before = model.consequent_layer.theta_gates.detach().clone()

    model.convert_to_first_order()
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_dg_aletsk_regressor_convert_to_first_order_preserves_theta_values() -> None:
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    theta_before = model.consequent_layer.theta_gates.detach().clone()

    model.convert_to_first_order()
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_adatsk_classifier_consequent_batch_norm() -> None:
    model = AdaTSKClassifier(_build_input_mfs(), n_classes=2, consequent_batch_norm=True)
    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,), dtype=torch.long)

    history = model.fit(x, y, epochs=2, learning_rate=1e-2, batch_size=4)
    assert history["stopped_epoch"] == 2
    assert model.predict(x).shape == (8,)


def test_adatsk_classifier_custom_rule_base_and_rules() -> None:
    input_mfs = _build_input_mfs(n_inputs=2, n_mfs=2)
    custom_rules = [(0, 0), (1, 1)]
    model = AdaTSKClassifier(
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
    model = AdaTSKRegressor(
        _build_input_mfs(n_inputs=2, n_mfs=2),
        consequent_batch_norm=True,
    )
    x = torch.randn(8, 2)
    y = torch.randn(8)
    history = model.fit(x, y, epochs=2, learning_rate=1e-2, batch_size=4)
    assert history["stopped_epoch"] == 2
    assert model.predict(x).shape == (8,)


def test_htsk_classifier_fit_history_keys_without_val() -> None:
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,), dtype=torch.long)

    history = model.fit(x, y, epochs=3)

    assert set(history.keys()) == {"train", "ur", "val", "stopped_epoch"}
    assert len(history["val"]) == 0
    assert history["stopped_epoch"] == 3


def test_htsk_classifier_early_stopping_with_val_data() -> None:
    torch.manual_seed(42)
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
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
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
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
        DombiTSKClassifier(_build_input_mfs(), n_classes=1, lambda_=1.0)


def test_dg_aletsk_classifier_fit_first_order_consequents_requires_conversion() -> None:
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
    x = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,), dtype=torch.long)

    with pytest.raises(ValueError, match=r"convert_to_first_order\(\) must be called before LSE consequent fitting"):
        model._fit_first_order_consequents_lse(x, y)


def test_dg_aletsk_classifier_search_thresholds_inplace_true_loads_state() -> None:
    torch.manual_seed(0)
    model = DGALETSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_init=1.0)
    model.convert_to_first_order()
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)

    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=True, verbose=False)

    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}
    assert isinstance(result["best_score"], float)


def test_dg_aletsk_regressor_convert_to_first_order_and_search_inplace_true() -> None:
    torch.manual_seed(0)
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]

    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)

    result = model.search_thresholds(x, y, x_val=x, y_val=y, use_lse=True, inplace=True)
    assert set(result.keys()) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dg_aletsk_regressor_fit_first_order_consequents_requires_conversion() -> None:
    model = DGALETSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_init=1.0)
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
    model = HTSKClassifier(
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
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
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
    """verbose=True + early stopping exercises logging lines 246 and 257."""
    torch.manual_seed(42)
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
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
        verbose=True,
    )
    assert history["stopped_epoch"] < 500


def test_htsk_classifier_fit_verbose_no_validation() -> None:
    """verbose=True without validation exercises the no-val logging path (line 261)."""
    torch.manual_seed(0)
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,), dtype=torch.long)

    history = model.fit(x, y, epochs=10, verbose=True)
    assert len(history["train"]) == 10


# ===========================================================================
# HTSKRegressor
# ===========================================================================


def test_htsk_regressor_init_validates_arguments() -> None:
    with pytest.raises(ValueError, match="input_mfs must not be empty"):
        HTSKRegressor({})


def test_htsk_regressor_forward_predict_shapes() -> None:
    model = HTSKRegressor(_build_input_mfs())
    x = torch.randn(8, 3)

    out = model.forward(x)
    pred = model.predict(x)

    assert out.shape == (8, 1)
    assert pred.shape == (8,)


def test_htsk_regressor_forward_antecedents_row_sum_one() -> None:
    model = HTSKRegressor(_build_input_mfs())
    x = torch.randn(6, 3)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_htsk_regressor_fit_returns_history() -> None:
    torch.manual_seed(1)
    model = HTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
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
    model = HTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(40, 2)
    y = x[:, 0] + 0.5 * x[:, 1]

    history = model.fit(x, y, epochs=50, learning_rate=1e-2)

    assert history["train"][-1] < history["train"][0]


def test_htsk_regressor_fit_validates_inputs() -> None:
    model = HTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
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
    model = HTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
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
    model = HTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(10, 2)
    y = torch.randn(10)

    with pytest.raises(ValueError, match="expected x_val shape"):
        model.fit(x, y, epochs=1, x_val=torch.randn(5, 3), y_val=torch.randn(5))

    with pytest.raises(ValueError, match="expected y_val shape"):
        model.fit(x, y, epochs=1, x_val=torch.randn(5, 2), y_val=torch.randn(5, 1))


def test_htsk_regressor_consequent_batch_norm() -> None:
    torch.manual_seed(1)
    model = HTSKRegressor(
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
    model = HTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
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
        verbose=True,
    )
    assert history["stopped_epoch"] < 2000


def test_htsk_regressor_fit_verbose_no_validation() -> None:
    torch.manual_seed(0)
    model = HTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)

    history = model.fit(x, y, epochs=10, verbose=True)
    assert len(history["train"]) == 10


def test_htsk_regressor_single_sample() -> None:
    torch.manual_seed(0)
    model = HTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(1, 2)
    y = torch.tensor([1.0])

    history = model.fit(x, y, epochs=3)
    assert len(history["train"]) == 3

    pred = model.predict(x)
    assert pred.shape == (1,)


def test_htsk_regressor_constant_targets() -> None:
    torch.manual_seed(0)
    model = HTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.full((20,), 3.14)

    _history = model.fit(x, y, epochs=200, learning_rate=5e-2)
    pred = model.predict(x)

    # Should converge close to the constant target
    assert float(torch.abs(pred.mean() - 3.14)) < 1.0
