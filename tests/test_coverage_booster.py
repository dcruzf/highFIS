"""Coverage boosting tests to cover all previously missed lines and branches in highFIS."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from highfis import (
    ADATSKClassifier,
    ADPTSKClassifier,
    DGALETSKClassifier,
    DGTSKClassifier,
    DGTSKRegressor,
    HTSKRegressor,
    LogTSKRegressor,
)
from highfis.estimators._adaptive import _validate_adptsk_paper_strict_input_range
from highfis.estimators._dg_aletsk import _validate_dg_aletsk_paper_strict_input_range
from highfis.estimators._dg_tsk import _validate_dg_tsk_paper_strict_input_range
from highfis.estimators._fsre import FSREADATSKClassifier, _validate_adatsk_paper_strict_input_range
from highfis.layers import GatedClassificationConsequentLayer, GatedRegressionConsequentLayer
from highfis.memberships import GaussianMF
from highfis.models import DGALETSKClassifierModel, DGTSKClassifierModel, DombiTSKClassifierModel

# ============================================================================
# 1. Empty array validations in _validate_*_paper_strict_input_range
# ============================================================================


def test_paper_strict_input_range_empty_arrays() -> None:
    # All of these should return early without error when passed empty arrays.
    empty = np.array([])
    _validate_adptsk_paper_strict_input_range(empty)
    _validate_dg_aletsk_paper_strict_input_range(empty)
    _validate_dg_tsk_paper_strict_input_range(empty)
    _validate_adatsk_paper_strict_input_range(empty)


# ============================================================================
# 2. Strict validation check with x_val in fit()
# ============================================================================


def test_fit_with_strict_and_validation_data() -> None:
    rng = np.random.default_rng(42)
    # Inputs must be in [0, 1] for paper_strict in these models
    x = rng.uniform(0.0, 1.0, (40, 2)).astype(np.float32)
    y = rng.choice([0, 1], size=40).astype(np.int64)
    x_val = rng.uniform(0.0, 1.0, (10, 2)).astype(np.float32)
    y_val = rng.choice([0, 1], size=10).astype(np.int64)

    # 2.1 ADPTSKClassifier
    clf_adp = ADPTSKClassifier(paper_strict=True, epochs=1)
    clf_adp.fit(x, y, x_val=x_val, y_val=y_val)

    # 2.2 ADATSKClassifier
    clf_ada = ADATSKClassifier(paper_strict=True, epochs=1)
    clf_ada.fit(x, y, x_val=x_val, y_val=y_val)

    # 2.3 DGALETSKClassifier (requires dg_epochs=10, finetune_epochs=50 in paper_strict)
    clf_dga = DGALETSKClassifier(paper_strict=True, dg_epochs=10, finetune_epochs=50)
    clf_dga.fit(x, y, x_val=x_val, y_val=y_val)

    # 2.4 DGTSKClassifier (requires dg_epochs=10 or 300, finetune_epochs=200 or 300 in paper_strict)
    clf_dgt = DGTSKClassifier(paper_strict=True, dg_epochs=10, finetune_epochs=200)
    clf_dgt.fit(x, y, x_val=x_val, y_val=y_val)

    # 2.5 FSREADATSKClassifier (requires fs_epochs=200, re_epochs=200, finetune_epochs=200 in paper_strict)
    # But wait, to keep tests fast, let's not run a 200 epochs fit in the test if we don't have to,
    # or we can pass a dummy dataset or run for 200 epochs on CPU which is very fast anyway
    clf_fsre = FSREADATSKClassifier(paper_strict=True, fs_epochs=200, re_epochs=200, finetune_epochs=200)
    clf_fsre.fit(x, y, x_val=x_val, y_val=y_val)


# ============================================================================
# 3. Validation errors for out-of-range inputs
# ============================================================================


def test_paper_strict_out_of_range_inputs() -> None:
    bad_x = np.array([[2.0, 0.5]])

    with pytest.raises(ValueError, match=" linearly normalized to \\[0,1\\]"):
        _validate_dg_aletsk_paper_strict_input_range(bad_x)

    with pytest.raises(ValueError, match=" linearly normalized to \\[0,1\\]"):
        _validate_dg_tsk_paper_strict_input_range(bad_x)


# ============================================================================
# 4. rule_base="pfrb" with pfrb_max_rules=None in DGALETSK
# ============================================================================


def test_dgaletsk_pfrb_with_no_max_rules() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, (20, 2)).astype(np.float32)
    y = rng.choice([0, 1], size=20).astype(np.int64)

    # Fits with pfrb and pfrb_max_rules=None to trigger resolve logic
    clf = DGALETSKClassifier(
        rule_base="pfrb",
        pfrb_max_rules=None,
        dg_epochs=1,
        finetune_epochs=1,
    )
    clf.fit(x, y)
    assert clf.pfrb_max_rules is None


# ============================================================================
# 5. HTSKRegressor & LogTSKRegressor _get_trainer in paper_strict
# ============================================================================


def test_regressors_strict_trainers() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((35, 2)).astype(np.float32)
    y = rng.standard_normal((35,)).astype(np.float32)

    # HTSKRegressor
    reg_htsk = HTSKRegressor(paper_strict=True)
    reg_htsk.fit(x, y)

    # LogTSKRegressor
    reg_log = LogTSKRegressor(paper_strict=True)
    reg_log.fit(x, y)


# ============================================================================
# 6. Yager strict-mode validation check for k <= 1.0
# ============================================================================


def test_yager_strict_rejects_small_k() -> None:
    from highfis import AYATSKClassifier

    with pytest.raises(ValueError, match=r"paper_strict requires k > 1\.0"):
        AYATSKClassifier(paper_strict=True, k=1.0)


# ============================================================================
# 7. FSRE-ADATSK zeta_theta checks for low and high dimensional data
# ============================================================================


def test_fsre_strict_zeta_theta_validation() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, (5, 2)).astype(np.float32)
    y = rng.choice([0, 1], size=5).astype(np.int64)

    # 7.1 Low dimensional (< 1000 features) with bad zeta_theta
    clf_low = FSREADATSKClassifier(
        paper_strict=True,
        zeta_theta=0.5,
        fs_epochs=200,
        re_epochs=200,
        finetune_epochs=200,
    )
    with pytest.raises(ValueError, match=r"paper_strict requires zeta_theta=0\.3 for low-dimensional data"):
        clf_low.fit(x, y)

    # 7.2 High dimensional (>= 1000 features) with bad zeta_theta
    # We trigger the validation by calling fit on high-dim x
    x_high = rng.uniform(0.0, 1.0, (5, 1005)).astype(np.float32)
    y_high = rng.choice([0, 1], size=5).astype(np.int64)

    # If we pass correct zeta_lambda=0.4 but bad zeta_theta=0.3
    clf = FSREADATSKClassifier(
        paper_strict=True,
        zeta_lambda=0.4,
        zeta_theta=0.3,
        fs_epochs=200,
        re_epochs=200,
        finetune_epochs=200,
    )
    with pytest.raises(ValueError, match=r"paper_strict requires zeta_theta=0\.5 for high-dimensional data"):
        clf.fit(x_high, y_high)


# ============================================================================
# 8. init_consequents_from_labels on already converted models
# ============================================================================


def test_init_consequents_raises_after_conversion() -> None:
    mfs = {"x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]}

    # 8.1 DGALETSKClassifierModel
    model_dga = DGALETSKClassifierModel(mfs, n_classes=2)
    model_dga.convert_to_first_order()
    with pytest.raises(ValueError, match="requires a zero-order consequent layer"):
        model_dga.init_consequents_from_labels(torch.tensor([0, 1]))

    # 8.2 DGTSKClassifierModel
    model_dg = DGTSKClassifierModel(mfs, n_classes=2)
    model_dg.convert_to_first_order()
    with pytest.raises(ValueError, match="requires a zero-order consequent layer"):
        model_dg.init_consequents_from_labels(torch.tensor([0, 1]))


# ============================================================================
# 9. DG-TSK Classifier Model search_thresholds top_rules selection (len(sr) < n_classes)
# ============================================================================


def test_dg_tsk_fit_top_rules_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    mfs = {
        "x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
        "x2": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    }
    # n_classes = 3, n_rules = 4
    model = DGTSKClassifierModel(mfs, n_classes=3, rule_base="cartesian")

    # Mock get_rule_gate_values to return zeros so that len(sr) naturally becomes 0 (< n_classes)
    monkeypatch.setattr(model, "get_rule_gate_values", lambda: torch.zeros(model.n_rules))

    # Run fit with inplace=True, structural=True, use_lse=True
    x = torch.randn(5, 2)
    y = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)

    result = model.search_thresholds(
        x,
        y,
        inplace=True,
        structural=True,
        use_lse=True,
        zeta_lambda=[0.0],
        zeta_theta=[0.5],
        verbose=False,
    )
    # The length of sr must be at least self.n_classes (3) due to top_k override
    assert len(result["surviving_rule_indices"]) >= 3


# ============================================================================
# 10. DombiTSKClassifierModel zero_consequent_init=False & missing params branch
# ============================================================================


def test_dombi_classifier_zero_init_branches() -> None:
    mfs = {"x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]}

    # 10.1 DombiTSKClassifierModel with zero_consequent_init=False
    model_false = DombiTSKClassifierModel(mfs, n_classes=2, zero_consequent_init=False)
    assert model_false.zero_consequent_init is False

    # 10.2 DombiTSKClassifierModel with mock consequent layer lacking weight/bias as Tensor
    class MockConsequent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = "not_a_tensor"
            self.bias = None

    model_mock = DombiTSKClassifierModel(mfs, n_classes=2, zero_consequent_init=True)
    # Replace the consequent layer with our mock
    model_mock.consequent_layer = MockConsequent()  # type: ignore[assignment]
    # Calling zero init should run without raising because the type checks fail gracefully
    model_mock._zero_initialize_consequents()


# ============================================================================
# 11. Checkpoint loading with first-order consequent mode
# ============================================================================


def test_dgtsk_persistence_with_first_order_consequent_mode(tmp_path: object) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (20, 2)).astype(np.float32)
    y = rng.choice([0, 1], size=20).astype(np.int64)

    path = str(tmp_path) + "/dgtsk_first_order.pt"  # type: ignore[operator]

    # 11.1 DGTSKClassifier
    clf = DGTSKClassifier(
        n_mfs=2,
        dg_epochs=1,
        finetune_epochs=1,
        use_lse=True,
        random_state=0,
    )
    clf.fit(X, y)
    clf.save(path)

    loaded = DGTSKClassifier.load(path)
    assert isinstance(loaded.model_.consequent_layer, GatedClassificationConsequentLayer)
    assert loaded.n_features_in_ == clf.n_features_in_
    assert np.array_equal(loaded.predict(X), clf.predict(X))

    # 11.2 DGTSKRegressor
    y_reg = rng.standard_normal((20,)).astype(np.float32)
    reg = DGTSKRegressor(
        n_mfs=2,
        dg_epochs=1,
        finetune_epochs=1,
        use_lse=True,
        random_state=0,
    )
    reg.fit(X, y_reg)
    reg.save(path)

    loaded_reg = DGTSKRegressor.load(path)
    assert isinstance(loaded_reg.model_.consequent_layer, GatedRegressionConsequentLayer)
    assert loaded_reg.n_features_in_ == reg.n_features_in_
    assert np.allclose(loaded_reg.predict(X), reg.predict(X), atol=1e-6)


# ============================================================================
# 12. FSRE-ADATSK Classifier FS/RE Epoch validation triggers
# ============================================================================


def test_fsre_epochs_strict_validation() -> None:
    with pytest.raises(ValueError, match="paper_strict requires fs_epochs=200"):
        FSREADATSKClassifier(paper_strict=True, fs_epochs=5)


# ============================================================================
# 13. FSRE-ADATSK Classifier invalid use_en_frb strict validation trigger
# ============================================================================


def test_fsre_invalid_en_frb_strict_validation() -> None:
    # Pass a string value disguised to bypass simple type checking, to hit the exception
    # (since normally it checks True/False, but we pass "invalid")
    with pytest.raises(ValueError, match="paper_strict requires use_en_frb=True"):
        FSREADATSKClassifier(paper_strict=True, use_en_frb="invalid")  # type: ignore
