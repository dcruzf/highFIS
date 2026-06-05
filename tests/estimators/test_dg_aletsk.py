from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from sklearn.pipeline import Pipeline

from highfis import DGALETSKClassifier, DGALETSKRegressor
from highfis.estimators._dg_aletsk import _validate_dg_aletsk_paper_strict_input_range


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return x, y


def test_dgaletsk_classifier_estimator_fit_three_phase_history() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    y = (rng.random(40) > 0.5).astype(int)
    clf = DGALETSKClassifier(
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


def test_dgaletsk_regressor_estimator_fit_three_phase_history() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    y = rng.standard_normal(40).astype(np.float32)
    reg = DGALETSKRegressor(
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


def test_dgaletsk_classifier_new_params_in_get_params() -> None:
    clf = DGALETSKClassifier(n_mfs=3, dg_epochs=15, finetune_epochs=50, use_lse=False)
    params = clf.get_params()
    assert params["dg_epochs"] == 15
    assert params["finetune_epochs"] == 50
    assert params["use_lse"] is False


def test_dgaletsk_classifier_paper_strict_defaults() -> None:
    clf = DGALETSKClassifier(paper_strict=True)
    assert clf.dg_epochs == 10
    assert clf.finetune_epochs == 50
    assert clf.learning_rate == 0.01
    assert clf.rule_base == "pfrb"
    assert clf.zeta_lambda == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    assert clf.zeta_theta == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def test_dgaletsk_regressor_no_paper_strict_support() -> None:
    with pytest.raises(TypeError):
        kwargs: dict[str, Any] = {"paper_strict": True}
        DGALETSKRegressor(**kwargs)


def test_dgaletsk_classifier_paper_strict_invalid_raises() -> None:
    with pytest.raises(ValueError, match="paper_strict requires dg_epochs=10"):
        DGALETSKClassifier(dg_epochs=20, paper_strict=True)

    with pytest.raises(ValueError, match="paper_strict requires finetune_epochs=50"):
        DGALETSKClassifier(finetune_epochs=100, paper_strict=True)

    with pytest.raises(ValueError, match="paper_strict requires learning_rate=1e-2"):
        DGALETSKClassifier(learning_rate=0.05, paper_strict=True)

    with pytest.raises(ValueError, match="paper_strict requires rule_base='pfrb'"):
        DGALETSKClassifier(rule_base="coco", paper_strict=True)

    with pytest.raises(ValueError, match="paper_strict requires zeta_lambda"):
        DGALETSKClassifier(zeta_lambda=[0.1], paper_strict=True)

    with pytest.raises(ValueError, match="paper_strict requires zeta_theta"):
        DGALETSKClassifier(zeta_theta=[0.1], paper_strict=True)


def test_dgaletsk_classifier_paper_strict_input_range_raises() -> None:
    X_invalid = np.array([[1.5, 0.2], [0.3, 0.8]])
    y = np.array([0, 1])

    clf = DGALETSKClassifier(paper_strict=True)
    with pytest.raises(ValueError, match="to be linearly normalized to"):
        clf.fit(X_invalid, y)


# --- Additional Estimator tests from test_estimators.py ---


def test_dgaletsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(40)
    est = DGALETSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=1.5,
        dg_epochs=2,
        finetune_epochs=1,
        zeta_lambda=[0.5],
        zeta_theta=[0.5],
        use_lse=False,
        structural_pruning=False,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    proba = est.predict_proba(x)
    pred = est.predict(x)
    score = est.score(x, y)

    assert proba.shape == (x.shape[0], 2)
    assert pred.shape == (x.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert 0.0 <= score <= 1.0


def test_dgaletsk_classifier_default_profile_is_paper_strict_like() -> None:
    est = DGALETSKClassifier()

    assert est.rule_base == "pfrb"
    assert est.finetune_epochs == 50
    assert est.use_lse is False
    assert est.zeta_lambda == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    assert est.zeta_theta == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def test_dgaletsk_classifier_default_pfrb_max_rules_policy() -> None:
    est = DGALETSKClassifier(rule_base="pfrb", pfrb_max_rules=None)

    x_low_dim = np.random.default_rng(1).normal(size=(2, 8)).astype(np.float32)
    input_mfs_low, _, _ = est._build_input_mfs(x_low_dim)
    assert len(input_mfs_low["x1"]) == 2

    x_high_dim = np.random.default_rng(2).normal(size=(2, 10_000)).astype(np.float32)
    input_mfs_high, _, _ = est._build_input_mfs(x_high_dim)
    assert len(input_mfs_high["x1"]) == 2


def test_dgaletsk_classifier_pre_train_hook_initializes_pfrb_consequents_from_labels() -> None:
    x, y = _make_dataset(30)
    est = DGALETSKClassifier(rule_base="pfrb", pfrb_max_rules=None)
    input_mfs, _, rule_base = est._build_input_mfs(x)
    model = cast(Any, est._build_model(input_mfs, n_classes=2, rule_base=rule_base))

    assert torch.allclose(model.consequent_layer.bias, torch.zeros_like(model.consequent_layer.bias))

    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    est._pre_train_hook(model, x_t, y_t)

    one_hot = model.consequent_layer.bias.detach().cpu().numpy()
    assert np.allclose(one_hot.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(one_hot, axis=1), y)


def test_dgaletsk_regressor_estimator_fit_predict_score() -> None:
    x = np.random.default_rng(123).normal(size=(40, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1] + 0.1 * np.random.default_rng(123).normal(size=40).astype(np.float32)
    est = DGALETSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=1.5,
        dg_epochs=2,
        finetune_epochs=1,
        zeta_lambda=[0.5],
        zeta_theta=[0.5],
        use_lse=False,
        structural_pruning=False,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    pred = est.predict(x)
    score = est.score(x, y)

    assert pred.shape == (x.shape[0],)
    assert isinstance(score, float)


def test_dgaletsk_classifier_estimator_pipeline_integration() -> None:
    x, y = _make_dataset(40)
    est = DGALETSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=1.5,
        dg_epochs=2,
        finetune_epochs=1,
        zeta_lambda=[0.5],
        zeta_theta=[0.5],
        use_lse=False,
        structural_pruning=False,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    pipe = Pipeline([("model", est)])
    pipe.fit(x, y)
    pred = pipe.predict(x[:10])

    assert pred.shape == (10,)


def test_paper_strict_input_range_empty_arrays_dg_aletsk() -> None:
    empty = np.array([])
    _validate_dg_aletsk_paper_strict_input_range(empty)


def test_fit_with_strict_and_validation_data_dg_aletsk() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, (40, 2)).astype(np.float32)
    y = rng.choice([0, 1], size=40).astype(np.int64)
    x_val = rng.uniform(0.0, 1.0, (10, 2)).astype(np.float32)
    y_val = rng.choice([0, 1], size=10).astype(np.int64)

    clf_dga = DGALETSKClassifier(paper_strict=True, dg_epochs=10, finetune_epochs=50)
    clf_dga.fit(x, y, x_val=x_val, y_val=y_val)


def test_paper_strict_out_of_range_inputs_dg_aletsk() -> None:
    bad_x = np.array([[2.0, 0.5]])
    with pytest.raises(ValueError, match=" linearly normalized to \\[0,1\\]"):
        _validate_dg_aletsk_paper_strict_input_range(bad_x)


def test_dgaletsk_pfrb_with_no_max_rules() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, (20, 2)).astype(np.float32)
    y = rng.choice([0, 1], size=20).astype(np.int64)

    clf = DGALETSKClassifier(
        rule_base="pfrb",
        pfrb_max_rules=None,
        dg_epochs=1,
        finetune_epochs=1,
    )
    clf.fit(x, y)
    assert clf.pfrb_max_rules is None
