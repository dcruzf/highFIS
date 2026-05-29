from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from highfis.layers import AdaSoftminRuleLayer
from highfis.memberships import GaussianMF
from highfis.models import FSREADATSKClassifierModel, FSREADATSKRegressorModel


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_fsre_adatsk_classifier_forward_shapes() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(5, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (5, 3)
    assert proba.shape == (5, 3)
    assert pred.shape == (5,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(5), atol=1e-6)


def test_fsre_adatsk_regressor_forward_shape() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)

    output = model.forward(x)

    assert output.shape == (4, 1)


def test_fsre_adatsk_forward_antecedents_row_sum_one() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(6, 2)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_fsre_adatsk_expand_to_en_frb_increases_rule_count() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    initial_rules = model.n_rules

    model.expand_to_en_frb()

    assert model.n_rules > initial_rules
    assert isinstance(model.rule_layer, AdaSoftminRuleLayer)


# ---------------------------------------------------------------------------
# lambda_gates shape — per-feature (n_inputs,), not per-rule-feature
# ---------------------------------------------------------------------------


def test_fsre_adatsk_classifier_lambda_gates_shared_per_feature() -> None:
    n_inputs = 4
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=n_inputs, n_mfs=2), n_classes=2)
    lam = model.consequent_layer.lambda_gates
    assert lam.shape == (n_inputs,), f"expected ({n_inputs},), got {lam.shape}"


def test_fsre_adatsk_regressor_lambda_gates_shared_per_feature() -> None:
    n_inputs = 3
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=n_inputs, n_mfs=2))
    lam = model.consequent_layer.lambda_gates
    assert lam.shape == (n_inputs,), f"expected ({n_inputs},), got {lam.shape}"


# ---------------------------------------------------------------------------
# mode attribute — initial and after each training phase
# ---------------------------------------------------------------------------


def test_fsre_adatsk_classifier_initial_mode_is_fs() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(), n_classes=2)
    assert model.consequent_layer.mode == "fs"


def test_fsre_adatsk_regressor_initial_mode_is_fs() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs())
    assert model.consequent_layer.mode == "fs"


def test_fsre_adatsk_classifier_expand_to_en_frb_resets_mode_to_fs() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    model.expand_to_en_frb()
    assert model.consequent_layer.mode == "fs"


# ---------------------------------------------------------------------------
# phase-specific gate activation: forward output differs between modes
# ---------------------------------------------------------------------------


def test_fsre_adatsk_classifier_mode_fs_uses_only_feature_gates() -> None:
    """FS mode: lambda_gates affect output; zeroing theta_gates has no effect."""
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "fs"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "FS mode must ignore theta_gates"


def test_fsre_adatsk_classifier_mode_re_uses_only_rule_gates() -> None:
    """RE mode: theta_gates affect output; zeroing lambda_gates has no effect."""
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "re"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "RE mode must ignore lambda_gates"


def test_fsre_adatsk_classifier_mode_finetune_ignores_all_gates() -> None:
    """Finetune mode: neither gate family affects output."""
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "finetune"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "Finetune mode must ignore all gates"


# ---------------------------------------------------------------------------
# get_feature_gate_values / get_rule_gate_values
# ---------------------------------------------------------------------------


def test_get_feature_gate_values_shape_classifier() -> None:
    n_inputs = 4
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=n_inputs, n_mfs=2), n_classes=2)
    vals = model.get_feature_gate_values()
    assert vals.shape == (n_inputs,)
    assert not vals.requires_grad


def test_get_rule_gate_values_shape_regressor() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=3, n_mfs=2))
    model.expand_to_en_frb()
    vals = model.get_rule_gate_values()
    assert vals.shape == (model.n_rules,)
    assert not vals.requires_grad


# ---------------------------------------------------------------------------
# prune_to_features
# ---------------------------------------------------------------------------


def test_prune_to_features_updates_model_attributes_classifier() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=4, n_mfs=2), n_classes=2)
    model.prune_to_features([0, 2])
    assert model.n_inputs == 2
    assert model.input_names == ["x1", "x3"]
    assert list(model.input_mfs.keys()) == ["x1", "x3"]


def test_prune_to_features_updates_model_attributes_regressor() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=4, n_mfs=2))
    model.prune_to_features([1, 3])
    assert model.n_inputs == 2
    assert model.input_names == ["x2", "x4"]


def test_prune_to_features_empty_raises_classifier() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="must not be empty"):
        model.prune_to_features([])


def test_prune_to_features_then_expand_en_frb_uses_surviving_features() -> None:
    """After prune+expand, the rule layer reflects the reduced feature set."""
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=4, n_mfs=2), n_classes=2)
    model.prune_to_features([0, 1])
    assert model.n_inputs == 2
    model.expand_to_en_frb()
    assert model.n_rules == 2 * 2  # n_inputs * n_mfs for En-FRB linear rule base
    x = torch.randn(5, 2)
    assert model.forward(x).shape == (5, 2)


# ---------------------------------------------------------------------------
# prune_to_rules
# ---------------------------------------------------------------------------


def test_prune_to_rules_classifier_updates_n_rules() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=3, n_mfs=2), n_classes=2)
    model.expand_to_en_frb()
    n_before = model.n_rules
    model.prune_to_rules([0, 1])
    assert model.n_rules == 2
    assert model.n_rules < n_before
    assert model.consequent_layer.mode == "finetune"


def test_prune_to_rules_regressor_copies_bias() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.expand_to_en_frb()
    old_bias = model.consequent_layer.bias.data.clone()
    surviving = [0, 2]
    model.prune_to_rules(surviving)
    assert torch.allclose(model.consequent_layer.bias.data, old_bias[surviving])


def test_prune_to_rules_empty_raises_regressor() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs())
    model.expand_to_en_frb()
    with pytest.raises(ValueError, match="must not be empty"):
        model.prune_to_rules([])


# ---------------------------------------------------------------------------
# FSRETrainer end-to-end
# ---------------------------------------------------------------------------


def test_fsre_trainer_classifier_end_to_end() -> None:
    from highfis.optim import FSRETrainer

    torch.manual_seed(0)
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=4, n_mfs=2), n_classes=2)
    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    trainer = FSRETrainer(fs_epochs=1, re_epochs=1, finetune_epochs=1)
    result = trainer.fit(model, x, y)

    assert "surviving_feature_indices" in result
    assert "surviving_rule_indices" in result
    assert "tau_lambda" in result
    assert "tau_theta" in result
    assert len(result["surviving_feature_indices"]) >= 1
    # After training the pruned model should still forward-pass
    sf = result["surviving_feature_indices"]
    assert model.forward(x[:, sf]).shape[1] == 2


def test_fsre_trainer_regressor_end_to_end() -> None:
    from highfis.optim import FSRETrainer

    torch.manual_seed(1)
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=3, n_mfs=2))
    x = torch.randn(15, 3)
    y = torch.randn(15)
    trainer = FSRETrainer(fs_epochs=1, re_epochs=1, finetune_epochs=1)
    result = trainer.fit(model, x, y)

    sf = result["surviving_feature_indices"]
    assert model.forward(x[:, sf]).shape == (15, 1)


def test_fsre_trainer_lower_bound_enforcement_classifier() -> None:
    """With zeta_theta=0.0 all rules fall below threshold; n_classes lower bound enforced."""
    from highfis.optim import FSRETrainer

    torch.manual_seed(42)
    n_classes = 3
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=3, n_mfs=2), n_classes=n_classes)
    x = torch.randn(20, 3)
    y = torch.randint(0, n_classes, (20,))
    # zeta_theta=0.0 → tau = max(M(θ)) → no rule strictly above threshold → lower bound kicks in
    trainer = FSRETrainer(
        fs_epochs=1,
        re_epochs=1,
        finetune_epochs=1,
        zeta_lambda=0.99,
        zeta_theta=0.0,
    )
    result = trainer.fit(model, x, y)
    assert len(result["surviving_rule_indices"]) >= n_classes


def test_fsre_trainer_no_structural_pruning_preserves_n_inputs() -> None:
    from highfis.optim import FSRETrainer

    torch.manual_seed(7)
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=4, n_mfs=2), n_classes=2)
    n_inputs_before = model.n_inputs
    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    trainer = FSRETrainer(
        fs_epochs=1,
        re_epochs=1,
        finetune_epochs=1,
        structural_pruning=False,
    )
    trainer.fit(model, x, y)
    assert model.n_inputs == n_inputs_before


# ---------------------------------------------------------------------------
# Estimator-level: _get_trainer returns FSRETrainer
# ---------------------------------------------------------------------------


def test_fsre_adatsk_classifier_estimator_default_uses_fsre_trainer() -> None:
    from highfis import FSREADATSKClassifier
    from highfis.optim import FSRETrainer

    clf = FSREADATSKClassifier()
    assert isinstance(clf._get_trainer(), FSRETrainer)


def test_fsre_adatsk_regressor_estimator_default_uses_fsre_trainer() -> None:
    from highfis import FSREADATSKRegressor
    from highfis.optim import FSRETrainer

    reg = FSREADATSKRegressor()
    assert isinstance(reg._get_trainer(), FSRETrainer)

    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "fs"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "FS mode must ignore theta_gates"


def test_fsre_adatsk_regressor_mode_re_uses_only_rule_gates() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "re"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "RE mode must ignore lambda_gates"


def test_fsre_adatsk_regressor_mode_finetune_ignores_all_gates() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)
    model.consequent_layer.mode = "finetune"

    with torch.no_grad():
        out_before = model.forward(x).clone()
        model.consequent_layer.lambda_gates.zero_()
        model.consequent_layer.theta_gates.zero_()
        out_after = model.forward(x)

    assert torch.allclose(out_before, out_after), "Finetune mode must ignore all gates"


# ---------------------------------------------------------------------------
# Coverage: batch_norm branch in prune_to_features (models/_fsre.py:175, 359)
# ---------------------------------------------------------------------------


def test_prune_to_features_with_consequent_bn_classifier() -> None:
    """prune_to_features updates consequent_bn when consequent_batch_norm=True."""
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=3, n_mfs=2), n_classes=2, consequent_batch_norm=True)
    assert model.consequent_batch_norm is True
    model.prune_to_features([0, 2])
    assert model.n_inputs == 2
    assert isinstance(model.consequent_bn, nn.BatchNorm1d)
    assert model.consequent_bn.num_features == 2


def test_prune_to_features_with_consequent_bn_regressor() -> None:
    """prune_to_features updates consequent_bn when consequent_batch_norm=True."""
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=3, n_mfs=2), consequent_batch_norm=True)
    assert model.consequent_batch_norm is True
    model.prune_to_features([1])
    assert model.n_inputs == 1
    assert isinstance(model.consequent_bn, nn.BatchNorm1d)
    assert model.consequent_bn.num_features == 1


# ---------------------------------------------------------------------------
# Coverage: empty raises in prune_to_features for regressor (models/_fsre.py:351)
# ---------------------------------------------------------------------------


def test_prune_to_features_empty_raises_regressor() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
    with pytest.raises(ValueError, match="surviving_features must not be empty"):
        model.prune_to_features([])


# ---------------------------------------------------------------------------
# Coverage: edge case - all feature gates ≤ tau → keep argmax (optim/_fsre.py:244)
# ---------------------------------------------------------------------------


def test_fsre_trainer_all_features_gated_out_keeps_one() -> None:
    """When all feature gates fall below tau_lambda, the top-1 feature is kept."""
    from highfis.optim import FSRETrainer

    torch.manual_seed(0)
    model = FSREADATSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,))
    # zeta_lambda=0.0 → tau = max gate value → no feature strictly above → edge case
    trainer = FSRETrainer(
        fs_epochs=1,
        re_epochs=1,
        finetune_epochs=1,
        zeta_lambda=0.0,
    )
    result = trainer.fit(model, x, y)
    assert len(result["surviving_feature_indices"]) == 1


# ---------------------------------------------------------------------------
# Coverage: estimator predict_proba / predict with wrong feature count
# ---------------------------------------------------------------------------


def test_fsre_adatsk_classifier_predict_proba_wrong_n_features() -> None:
    import numpy as np

    from highfis import FSREADATSKClassifier

    X = np.random.default_rng(0).standard_normal((20, 3))
    y = np.random.default_rng(0).integers(0, 2, size=20)
    clf = FSREADATSKClassifier(fs_epochs=1, re_epochs=1, finetune_epochs=1)
    clf.fit(X, y)
    with pytest.raises(ValueError, match="expected"):
        clf.predict_proba(X[:, :2])


def test_fsre_adatsk_regressor_predict_wrong_n_features() -> None:
    import numpy as np

    from highfis import FSREADATSKRegressor

    X = np.random.default_rng(1).standard_normal((20, 3))
    y = np.random.default_rng(1).standard_normal(20)
    reg = FSREADATSKRegressor(fs_epochs=1, re_epochs=1, finetune_epochs=1)
    reg.fit(X, y)
    with pytest.raises(ValueError, match="expected"):
        reg.predict(X[:, :2])
