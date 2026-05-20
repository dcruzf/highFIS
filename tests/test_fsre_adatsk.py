from __future__ import annotations

import torch

from highfis.layers import AdaSoftminRuleLayer
from highfis.memberships import GaussianMF
from highfis.models import FSREAdaTSKClassifierModel, FSREAdaTSKRegressorModel


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_fsre_adatsk_classifier_forward_shapes() -> None:
    model = FSREAdaTSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(5, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (5, 3)
    assert proba.shape == (5, 3)
    assert pred.shape == (5,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(5), atol=1e-6)


def test_fsre_adatsk_regressor_forward_shape() -> None:
    model = FSREAdaTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)

    output = model.forward(x)

    assert output.shape == (4, 1)


def test_fsre_adatsk_forward_antecedents_row_sum_one() -> None:
    model = FSREAdaTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(6, 2)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_fsre_adatsk_expand_to_en_frb_increases_rule_count() -> None:
    model = FSREAdaTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    initial_rules = model.n_rules

    model.expand_to_en_frb()

    assert model.n_rules > initial_rules
    assert isinstance(model.rule_layer, AdaSoftminRuleLayer)


# ---------------------------------------------------------------------------
# lambda_gates shape — per-feature (n_inputs,), not per-rule-feature
# ---------------------------------------------------------------------------


def test_fsre_adatsk_classifier_lambda_gates_shared_per_feature() -> None:
    n_inputs = 4
    model = FSREAdaTSKClassifierModel(_build_input_mfs(n_inputs=n_inputs, n_mfs=2), n_classes=2)
    lam = model.consequent_layer.lambda_gates
    assert lam.shape == (n_inputs,), f"expected ({n_inputs},), got {lam.shape}"


def test_fsre_adatsk_regressor_lambda_gates_shared_per_feature() -> None:
    n_inputs = 3
    model = FSREAdaTSKRegressorModel(_build_input_mfs(n_inputs=n_inputs, n_mfs=2))
    lam = model.consequent_layer.lambda_gates
    assert lam.shape == (n_inputs,), f"expected ({n_inputs},), got {lam.shape}"


# ---------------------------------------------------------------------------
# mode attribute — initial and after each training phase
# ---------------------------------------------------------------------------


def test_fsre_adatsk_classifier_initial_mode_is_fs() -> None:
    model = FSREAdaTSKClassifierModel(_build_input_mfs(), n_classes=2)
    assert model.consequent_layer.mode == "fs"


def test_fsre_adatsk_regressor_initial_mode_is_fs() -> None:
    model = FSREAdaTSKRegressorModel(_build_input_mfs())
    assert model.consequent_layer.mode == "fs"


def test_fsre_adatsk_classifier_expand_to_en_frb_resets_mode_to_fs() -> None:
    model = FSREAdaTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    model.expand_to_en_frb()
    assert model.consequent_layer.mode == "fs"


# ---------------------------------------------------------------------------
# phase-specific gate activation: forward output differs between modes
# ---------------------------------------------------------------------------


def test_fsre_adatsk_classifier_mode_fs_uses_only_feature_gates() -> None:
    """FS mode: lambda_gates affect output; zeroing theta_gates has no effect."""
    model = FSREAdaTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "fs"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "FS mode must ignore theta_gates"


def test_fsre_adatsk_classifier_mode_re_uses_only_rule_gates() -> None:
    """RE mode: theta_gates affect output; zeroing lambda_gates has no effect."""
    model = FSREAdaTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "re"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "RE mode must ignore lambda_gates"


def test_fsre_adatsk_classifier_mode_finetune_ignores_all_gates() -> None:
    """Finetune mode: neither gate family affects output."""
    model = FSREAdaTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "finetune"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "Finetune mode must ignore all gates"


def test_fsre_adatsk_regressor_mode_fs_uses_only_feature_gates() -> None:
    model = FSREAdaTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "fs"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "FS mode must ignore theta_gates"


def test_fsre_adatsk_regressor_mode_re_uses_only_rule_gates() -> None:
    model = FSREAdaTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "re"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "RE mode must ignore lambda_gates"


def test_fsre_adatsk_regressor_mode_finetune_ignores_all_gates() -> None:
    model = FSREAdaTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "finetune"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "Finetune mode must ignore all gates"
