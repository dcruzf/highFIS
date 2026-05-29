from __future__ import annotations

import numpy as np
import pytest
import torch

from highfis.estimators import DGTSKClassifier, DGTSKRegressor, InputConfig
from highfis.layers import GatedClassificationConsequentLayer, GatedRegressionConsequentLayer
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


def test_dgtsk_estimator_instantiation() -> None:
    clf = DGTSKClassifier(n_mfs=2, mf_init="kmeans", use_en_frb=True)
    reg = DGTSKRegressor(n_mfs=2, mf_init="kmeans", use_en_frb=True)

    assert clf is not None
    assert reg is not None


def test_dgtsk_classifier_apply_thresholds_invalid_thresholds_raises() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="thresholds must be finite"):
        model.apply_thresholds(float("nan"), 0.0)


def test_dgtsk_classifier_search_thresholds_verbose_and_inplace_true() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

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


def test_dgtsk_regressor_search_thresholds_verbose_and_inplace_true() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

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


def test_dgtsk_classifier_search_thresholds_returns_result() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=5, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(x, y, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], inplace=False)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}
    assert 0.0 <= result["best_score"] <= 1.0


def test_dgtsk_regressor_search_thresholds_returns_result() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)
    model.fit_dg_phase(x, y, epochs=5, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(x, y, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], inplace=False)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgtsk_classifier_search_thresholds_default_zeta_lists() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(x, y, zeta_lambda=None, zeta_theta=None, inplace=False)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgtsk_regressor_search_thresholds_default_zeta_lists() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(x, y, zeta_lambda=None, zeta_theta=None, inplace=False)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


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


def test_dgtsk_classifier_fit_first_order_consequents_requires_conversion() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,))

    with pytest.raises(ValueError, match=r"convert_to_first_order\(\) must be called before LSE consequent fitting"):
        model._fit_first_order_consequents_lse(x, y)


def test_dgtsk_regressor_fit_first_order_consequents_requires_conversion() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(8, 2)
    y = torch.randn(8)

    with pytest.raises(ValueError, match=r"convert_to_first_order\(\) must be called before LSE consequent fitting"):
        model._fit_first_order_consequents_lse(x, y)


def test_dgtsk_classifier_fit_dg_phase_and_finetune() -> None:
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))

    history_dg = model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)
    assert isinstance(history_dg, dict)

    history_ft = model.fit_finetune(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)
    assert isinstance(history_ft, dict)


def test_dgtsk_regressor_fit_dg_phase_and_finetune() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)

    history_dg = model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)
    assert isinstance(history_dg, dict)

    history_ft = model.fit_finetune(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)
    assert isinstance(history_ft, dict)


def test_dgtsk_classifier_lambda_gates_shape_is_per_feature() -> None:
    """lambda_gates must be (n_inputs,) — shared across all rules per the DG-TSK paper."""
    model = DGTSKClassifierModel(_build_input_mfs(n_inputs=4, n_mfs=2), n_classes=2)
    assert model.rule_layer.lambda_gates.shape == (4,)


def test_dgtsk_regressor_lambda_gates_shape_is_per_feature() -> None:
    """lambda_gates must be (n_inputs,) — shared across all rules per the DG-TSK paper."""
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=5, n_mfs=2))
    assert model.rule_layer.lambda_gates.shape == (5,)


def test_dgtsk_classifier_first_order_consequent_mode_is_re() -> None:
    """After convert_to_first_order(), consequent mode must be 're' (rule gates only)."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    assert model.consequent_layer.mode == "re"


def test_dgtsk_regressor_first_order_consequent_mode_is_re() -> None:
    """After convert_to_first_order(), consequent mode must be 're' (rule gates only)."""
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    assert model.consequent_layer.mode == "re"


# ---------------------------------------------------------------------------
# Paper-faithful DG phase: antecedents frozen (paper §3.3)
# ---------------------------------------------------------------------------


def test_dgtsk_classifier_fit_dg_phase_freezes_antecedents() -> None:
    """Antecedent MF parameters must not change during fit_dg_phase (paper §3.3)."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))

    snapshot = {name: p.detach().clone() for name, p in model.membership_layer.named_parameters()}
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    for name, p in model.membership_layer.named_parameters():
        assert torch.allclose(p.detach(), snapshot[name]), f"antecedent param '{name}' changed during fit_dg_phase"


def test_dgtsk_regressor_fit_dg_phase_freezes_antecedents() -> None:
    """Antecedent MF parameters must not change during fit_dg_phase (paper §3.3)."""
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(16, 2)
    y = torch.randn(16)

    snapshot = {name: p.detach().clone() for name, p in model.membership_layer.named_parameters()}
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    for name, p in model.membership_layer.named_parameters():
        assert torch.allclose(p.detach(), snapshot[name]), f"antecedent param '{name}' changed during fit_dg_phase"


def test_dgtsk_fit_dg_phase_restores_requires_grad_after_exception() -> None:
    """requires_grad must be restored even when fit() raises an exception."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)

    # Pass x with wrong number of features to provoke a runtime error
    with pytest.raises(Exception):  # noqa: B017
        model.fit_dg_phase(torch.randn(4, 99), torch.randint(0, 2, (4,)), epochs=1)

    for p in model.membership_layer.parameters():
        assert p.requires_grad, "requires_grad not restored after exception in fit_dg_phase"


# ---------------------------------------------------------------------------
# Paper-faithful finetune: consequents reset to zero (paper §3.3)
# ---------------------------------------------------------------------------


def test_dgtsk_classifier_fit_finetune_resets_consequents() -> None:
    """Consequent weight and bias must be zeroed before finetuning (paper §3.3)."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)

    # Force non-zero values to confirm they are overwritten
    torch.nn.init.constant_(model.consequent_layer.weight, 99.0)
    torch.nn.init.constant_(model.consequent_layer.bias, 99.0)

    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,))
    model.fit_finetune(x, y, epochs=0)

    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)
    assert model.consequent_layer.weight.abs().max().item() == 0.0
    assert model.consequent_layer.bias.abs().max().item() == 0.0


def test_dgtsk_regressor_fit_finetune_resets_consequents() -> None:
    """Consequent weight and bias must be zeroed before finetuning (paper §3.3)."""
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)

    torch.nn.init.constant_(model.consequent_layer.weight, 99.0)
    torch.nn.init.constant_(model.consequent_layer.bias, 99.0)

    x = torch.randn(8, 2)
    y = torch.randn(8)
    model.fit_finetune(x, y, epochs=0)

    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)
    assert model.consequent_layer.weight.abs().max().item() == 0.0
    assert model.consequent_layer.bias.abs().max().item() == 0.0


def test_dgtsk_classifier_fit_finetune_skips_reset_on_zero_order() -> None:
    """fit_finetune on a zero-order model must not raise and must not reset anything."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    # model is still zero-order — fit_finetune should just call fit without resetting
    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,))
    history = model.fit_finetune(x, y, epochs=1, batch_size=8, shuffle=False)
    assert "train" in history


# ---------------------------------------------------------------------------
# P-FRB one-hot initialisation (paper eq. 24)
# ---------------------------------------------------------------------------


def test_dgtsk_classifier_init_consequents_from_labels() -> None:
    """bias must equal one-hot encoding of y (paper eq. 24)."""
    mfs = {f"x{i}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(2)] for i in range(2)}
    model = DGTSKClassifierModel(mfs, n_classes=4, rule_base="coco")
    # coco with 2 MFs per input gives 2 rules; use 2 labels
    n_rules = model.n_rules
    y = torch.arange(min(n_rules, 4), dtype=torch.long)

    model.init_consequents_from_labels(y)

    expected = torch.zeros(n_rules, 4)
    n = min(len(y), n_rules)
    expected[:n].scatter_(1, y[:n].unsqueeze(1), 1.0)
    assert torch.allclose(model.consequent_layer.bias.data, expected)


def test_dgtsk_classifier_init_consequents_from_labels_partial_fill() -> None:
    """When n_samples < n_rules, remaining rows stay zero."""
    model = DGTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=3), n_classes=2)
    # provide only 1 label (fewer than n_rules)
    model.init_consequents_from_labels(torch.tensor([1], dtype=torch.long))

    assert model.consequent_layer.bias.data[0, 1].item() == 1.0
    assert model.consequent_layer.bias.data[1:].abs().sum().item() == 0.0


def test_dgtsk_classifier_init_consequents_from_labels_raises_on_first_order() -> None:
    """Calling init_consequents_from_labels after convert_to_first_order must raise."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()

    with pytest.raises(ValueError, match="zero-order consequent layer"):
        model.init_consequents_from_labels(torch.zeros(2, dtype=torch.long))


# ---------------------------------------------------------------------------
# Estimator-level tests
# ---------------------------------------------------------------------------


def test_dgtsk_classifier_estimator_fit_three_phase_history() -> None:
    """DGTSKClassifier.fit() must produce history_ with dg/threshold/finetune keys."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    y = (rng.random(40) > 0.5).astype(int)
    clf = DGTSKClassifier(
        n_mfs=2,
        dg_epochs=2,
        finetune_epochs=3,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    clf.fit(X, y)
    assert isinstance(clf.history_, dict)
    assert set(clf.history_) >= {"dg", "threshold", "finetune"}


def test_dgtsk_regressor_estimator_fit_three_phase_history() -> None:
    """DGTSKRegressor.fit() must produce history_ with dg/threshold/finetune keys."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    y = rng.standard_normal(40).astype(np.float32)
    reg = DGTSKRegressor(
        n_mfs=2,
        dg_epochs=2,
        finetune_epochs=3,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    reg.fit(X, y)
    assert isinstance(reg.history_, dict)
    assert set(reg.history_) >= {"dg", "threshold", "finetune"}


def test_dgtsk_classifier_estimator_with_gradient_trainer() -> None:
    """Passing GradientTrainer overrides 3-phase training (flat history dict)."""
    from highfis import GradientTrainer

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    y = (rng.random(20) > 0.5).astype(int)
    clf = DGTSKClassifier(n_mfs=2, dg_epochs=2, trainer=GradientTrainer(epochs=3), random_state=0)
    clf.fit(X, y)
    assert isinstance(clf.history_, dict)
    assert "dg" not in clf.history_


def test_dgtsk_classifier_new_params_in_get_params() -> None:
    clf = DGTSKClassifier(n_mfs=3, dg_epochs=15, finetune_epochs=50, use_lse=False)
    params = clf.get_params()
    assert params["dg_epochs"] == 15
    assert params["finetune_epochs"] == 50
    assert params["use_lse"] is False


def test_dgtsk_classifier_pfrb_pre_train_hook() -> None:
    """With rule_base='pfrb', _pre_train_hook initializes consequents from labels."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    y = (rng.random(20) > 0.5).astype(int)
    clf = DGTSKClassifier(
        n_mfs=2,
        rule_base="pfrb",
        dg_epochs=2,
        finetune_epochs=2,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    clf.fit(X, y)
    assert clf.rule_base_ == "coco"


# ---------------------------------------------------------------------------
# _build_optimizer: custom passthrough and adamw fallback
# ---------------------------------------------------------------------------


def test_dgtsk_classifier_build_optimizer_passthrough() -> None:
    """_build_optimizer returns a pre-built optimizer unchanged."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    custom_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = model._build_optimizer(custom_opt, 1e-3, 0.0)
    assert result is custom_opt


def test_dgtsk_classifier_build_optimizer_adamw_fallback() -> None:
    """optimizer_type='adamw' delegates to the base AdamW builder."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2, optimizer_type="adamw")
    result = model._build_optimizer(None, 1e-3, 1e-4)
    assert isinstance(result, torch.optim.AdamW)


def test_dgtsk_regressor_build_optimizer_passthrough() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    custom_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = model._build_optimizer(custom_opt, 1e-3, 0.0)
    assert result is custom_opt


def test_dgtsk_regressor_build_optimizer_adamw_fallback() -> None:
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2), optimizer_type="adamw")
    result = model._build_optimizer(None, 1e-3, 1e-4)
    assert isinstance(result, torch.optim.AdamW)


# ---------------------------------------------------------------------------
# prune_structure: input validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# search_thresholds: inplace=True, structural=False
# ---------------------------------------------------------------------------


def test_dgtsk_classifier_search_thresholds_non_structural_lse() -> None:
    """inplace=True, structural=False, use_lse=True: restore best state after LSE."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        inplace=True,
        structural=False,
        use_lse=True,
    )
    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta"}


def test_dgtsk_classifier_search_thresholds_non_structural_no_lse() -> None:
    """inplace=True, structural=False, use_lse=False: apply thresholds in-place."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        inplace=True,
        structural=False,
        use_lse=False,
    )
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta"}


def test_dgtsk_regressor_search_thresholds_non_structural_lse() -> None:
    """inplace=True, structural=False, use_lse=True for regressor."""
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        inplace=True,
        structural=False,
        use_lse=True,
    )
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta"}


def test_dgtsk_regressor_search_thresholds_non_structural_no_lse() -> None:
    """inplace=True, structural=False, use_lse=False for regressor."""
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        inplace=True,
        structural=False,
        use_lse=False,
    )
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta"}


# ---------------------------------------------------------------------------
# search_thresholds: sr fallback and sf non-empty (no fallback)
# ---------------------------------------------------------------------------


def test_dgtsk_classifier_search_thresholds_sr_fallback() -> None:
    """All rule theta-gates == 0 with zeta_theta=0 → sr=[] → fallback to all rules."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.consequent_layer.theta_gates.data.fill_(0.0)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    n_rules_orig = model.n_rules

    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[0.0],
        zeta_theta=[0.0],
        inplace=True,
        structural=True,
        use_lse=False,
    )
    assert result["surviving_rule_indices"] == list(range(n_rules_orig))


def test_dgtsk_regressor_search_thresholds_sf_non_empty_no_fallback() -> None:
    """Feature 0 gate > tau → sf=[0] is non-empty, no sf-fallback taken."""
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.rule_layer.lambda_gates.data = torch.tensor([1.0, 0.0])
    model.consequent_layer.theta_gates.data.fill_(1.0)
    x = torch.randn(16, 2)
    y = torch.randn(16)

    # zeta=1.0 → tau=min_gate=0 → gate 0 > 0 → sf=[0] (no fallback)
    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[1.0],
        zeta_theta=[1.0],
        inplace=True,
        structural=True,
        use_lse=False,
    )
    assert result["surviving_feature_indices"] == [0]


# ---------------------------------------------------------------------------
# fit_finetune: freeze_antecedents=False
# ---------------------------------------------------------------------------


def test_dgtsk_classifier_fit_finetune_unfreeze_antecedents() -> None:
    """freeze_antecedents=False sets consequent mode to 'finetune' during training."""
    model = DGTSKClassifierModel(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,))

    history = model.fit_finetune(x, y, epochs=1, batch_size=8, shuffle=False, freeze_antecedents=False)
    assert isinstance(history, dict)
    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)


def test_dgtsk_regressor_fit_finetune_unfreeze_antecedents() -> None:
    """freeze_antecedents=False sets consequent mode to 'finetune' during training."""
    model = DGTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    x = torch.randn(8, 2)
    y = torch.randn(8)

    history = model.fit_finetune(x, y, epochs=1, batch_size=8, shuffle=False, freeze_antecedents=False)
    assert isinstance(history, dict)
    assert isinstance(model.consequent_layer, GatedRegressionConsequentLayer)


# ---------------------------------------------------------------------------
# DGTrainer: feature pruning slices x_ft and x_val_ft (lines 228-229 _dg.py)
# ---------------------------------------------------------------------------


def test_dg_trainer_slices_x_ft_when_features_pruned() -> None:
    """DGTrainer slices x and x_val to surviving features after structural pruning."""
    from highfis.optim import DGTrainer

    model = DGTSKClassifierModel(_build_input_mfs(n_inputs=3, n_mfs=2), n_classes=2)
    # Feature 0 dominant; features 1 and 2 will be pruned by zeta=1.0
    model.rule_layer.lambda_gates.data = torch.tensor([1.0, 0.0, 0.0])
    model.consequent_layer.theta_gates.data.fill_(1.0)

    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    x_val = torch.randn(8, 3)
    y_val = torch.randint(0, 2, (8,))

    trainer = DGTrainer(
        dg_epochs=0,
        finetune_epochs=1,
        zeta_lambda=[1.0],
        zeta_theta=[1.0],
        structural_pruning=True,
        use_lse=False,
        dg_patience=None,
        finetune_patience=None,
        finetune_restore_best=False,
    )
    history = trainer.fit(model, x, y, x_val=x_val, y_val=y_val)
    assert set(history) == {"dg", "threshold", "finetune"}
    assert history["threshold"]["surviving_feature_indices"] == [0]


# ---------------------------------------------------------------------------
# Persistence: DGTSKClassifier save / load
# ---------------------------------------------------------------------------


def test_dgtsk_classifier_save_load_roundtrip(tmp_path: object) -> None:
    """DGTSKClassifier.save() and .load() produce an identical predictor."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    clf = DGTSKClassifier(
        n_mfs=2,
        dg_epochs=1,
        finetune_epochs=1,
        structural_pruning=False,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    clf.fit(X, y)
    path = str(tmp_path) + "/dgtsk_clf.pt"  # type: ignore[operator]
    clf.save(path)

    loaded = DGTSKClassifier.load(path)
    assert loaded.n_features_in_ == clf.n_features_in_
    assert np.array_equal(loaded.classes_, clf.classes_)
    assert np.array_equal(loaded.predict(X), clf.predict(X))


def test_dgtsk_classifier_load_with_input_configs(tmp_path: object) -> None:
    """DGTSKClassifier.load() restores InputConfig objects from the checkpoint."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    configs = [InputConfig(name="a", n_mfs=2), InputConfig(name="b", n_mfs=2)]

    clf = DGTSKClassifier(
        input_configs=configs,
        dg_epochs=1,
        finetune_epochs=1,
        structural_pruning=False,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    clf.fit(X, y)
    path = str(tmp_path) + "/dgtsk_clf_cfg.pt"  # type: ignore[operator]
    clf.save(path)

    loaded = DGTSKClassifier.load(path)
    assert loaded.n_features_in_ == clf.n_features_in_
    assert np.array_equal(loaded.predict(X), clf.predict(X))


# ---------------------------------------------------------------------------
# Persistence: DGTSKRegressor save / load
# ---------------------------------------------------------------------------


def test_dgtsk_regressor_save_load_roundtrip(tmp_path: object) -> None:
    """DGTSKRegressor.save() and .load() produce an identical predictor."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    y = (X[:, 0] * 2.0).astype(np.float32)

    reg = DGTSKRegressor(
        n_mfs=2,
        dg_epochs=1,
        finetune_epochs=1,
        structural_pruning=False,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    reg.fit(X, y)
    path = str(tmp_path) + "/dgtsk_reg.pt"  # type: ignore[operator]
    reg.save(path)

    loaded = DGTSKRegressor.load(path)
    assert loaded.n_features_in_ == reg.n_features_in_
    assert np.allclose(loaded.predict(X), reg.predict(X), atol=1e-5)


def test_dgtsk_regressor_load_with_input_configs(tmp_path: object) -> None:
    """DGTSKRegressor.load() restores InputConfig objects from the checkpoint."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    y = X[:, 0] + X[:, 1]
    configs = [InputConfig(name="u", n_mfs=2), InputConfig(name="v", n_mfs=2)]

    reg = DGTSKRegressor(
        input_configs=configs,
        dg_epochs=1,
        finetune_epochs=1,
        structural_pruning=False,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    reg.fit(X, y)
    path = str(tmp_path) + "/dgtsk_reg_cfg.pt"  # type: ignore[operator]
    reg.save(path)

    loaded = DGTSKRegressor.load(path)
    assert loaded.n_features_in_ == reg.n_features_in_


def test_dgtsk_regressor_load_restores_first_order_architecture(tmp_path: object) -> None:
    """DGTSKRegressor.load() converts model to first-order when saved after finetuning."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    y = X[:, 0].astype(np.float32)

    reg = DGTSKRegressor(
        n_mfs=2,
        dg_epochs=1,
        finetune_epochs=1,
        use_lse=True,
        structural_pruning=False,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    reg.fit(X, y)  # use_lse=True + finetune → model is first-order after fit
    path = str(tmp_path) + "/dgtsk_reg_fo.pt"  # type: ignore[operator]
    reg.save(path)

    loaded = DGTSKRegressor.load(path)
    assert loaded.n_features_in_ == reg.n_features_in_
    assert np.allclose(loaded.predict(X), reg.predict(X), atol=1e-4)
