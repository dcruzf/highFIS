from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from highfis import (
    ADATSKClassifier,
    ADATSKRegressor,
    ADMTSKClassifier,
    ADMTSKRegressor,
    ADPTSKClassifier,
    ADPTSKRegressor,
)
from highfis.estimators import InputConfig
from highfis.estimators._adaptive import (
    _set_sigma_to_one_and_freeze,
    _validate_adptsk_paper_strict_input_range,
    _wrap_adatsk_gaussian_input_mfs,
)
from highfis.estimators._fsre import _validate_adatsk_paper_strict_input_range
from highfis.memberships import ADATSKGaussianMF, CompositeGaussianMF, GaussianMF, GaussianPiMF


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return x, y


def _make_regression_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(456)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2]).astype(np.float32)
    return x, y


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


def test_adatsk_classifier_paper_strict_defaults() -> None:
    clf = ADATSKClassifier(paper_strict=True)
    assert clf.n_mfs == 3
    assert clf.mf_init == "grid"
    assert clf.sigma_scale == 1.0
    assert clf.rule_base == "coco"
    assert clf.epochs == 200
    assert clf.learning_rate == 1e-2
    assert clf.batch_size is None


def test_adatsk_classifier_paper_strict_overrides_raise() -> None:
    with pytest.raises(ValueError, match="paper_strict requires n_mfs=3"):
        ADATSKClassifier(paper_strict=True, n_mfs=5)
    with pytest.raises(ValueError, match="paper_strict requires mf_init='grid'"):
        ADATSKClassifier(paper_strict=True, mf_init="kmeans")
    with pytest.raises(ValueError, match=r"paper_strict requires sigma_scale=1\.0"):
        ADATSKClassifier(paper_strict=True, sigma_scale=0.5)
    with pytest.raises(ValueError, match="paper_strict requires rule_base='coco'"):
        ADATSKClassifier(paper_strict=True, rule_base="cartesian")
    with pytest.raises(ValueError, match="paper_strict requires epochs=200"):
        ADATSKClassifier(paper_strict=True, epochs=100)
    with pytest.raises(ValueError, match=r"paper_strict requires learning_rate=1e-2"):
        ADATSKClassifier(paper_strict=True, learning_rate=1e-3)
    with pytest.raises(ValueError, match="paper_strict requires batch_size=None"):
        ADATSKClassifier(paper_strict=True, batch_size=10)


def test_adatsk_classifier_paper_strict_input_range() -> None:
    clf = ADATSKClassifier(paper_strict=True)
    x_bad = np.array([[-0.1, 0.5], [1.1, 0.5]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="paper_strict requires x to be linearly normalized to"):
        clf.fit(x_bad, y)

    x_good = np.array([[0.0, 0.5], [1.0, 0.5]])
    clf.fit(x_good, y)
    assert clf.paper_strict is True


def test_adatsk_regressor_no_paper_strict_support() -> None:
    with pytest.raises(TypeError):
        cast(Any, ADATSKRegressor)(paper_strict=True)


def test_composite_gaussian_mf_lower_bound() -> None:
    mf = CompositeGaussianMF(mean=0.0, sigma=1.0, eps=0.05)
    x = torch.tensor([-5.0, 0.0, 5.0])

    values = mf(x)

    assert torch.all(values >= 0.05)
    assert torch.all(values <= 1.0)


# --- ADATSK Estimator Tests ---


def test_adatsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = ADATSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
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


def test_adatsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = ADATSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


# --- ADPTSK Estimator Tests ---


def test_adptsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = ADPTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
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


def test_adptsk_classifier_default_estimator_uses_paper_rule_base_and_rule_count() -> None:
    x, y = _make_dataset(80)
    est = ADPTSKClassifier(epochs=1, random_state=7)
    est.fit(x, y)
    assert est.rule_base_ == "coco"
    assert est.model_.n_rules == 3


def test_adptsk_classifier_default_estimator_uses_paper_centers_and_sigma() -> None:
    x, y = _make_dataset(80)
    est = ADPTSKClassifier(epochs=1, random_state=7)
    est.fit(x, y)

    mfs = est.model_.input_mfs["x1"]
    assert len(mfs) == 3
    assert isinstance(mfs[0], GaussianPiMF)
    means = [float(cast(Tensor, cast(GaussianPiMF, mf).mean).detach().item()) for mf in mfs]
    sigmas = [float(cast(GaussianPiMF, mf).sigma.detach().item()) for mf in mfs]
    assert means == pytest.approx([0.0, 0.5, 1.0], abs=1e-8)
    assert sigmas == pytest.approx([1.0, 1.0, 1.0], abs=1e-8)


def test_adptsk_default_batch_size_policy() -> None:
    est = ADPTSKClassifier()
    assert est._resolve_default_batch_size(499) is None
    assert est._resolve_default_batch_size(500) == 100


def test_adptsk_classifier_paper_strict_uses_paper_protocol_defaults() -> None:
    est = ADPTSKClassifier(paper_strict=True)
    assert est.n_mfs == 3
    assert est.mf_init == "grid"
    assert est.sigma_scale == 1.0
    assert est.rule_base == "coco"
    assert est.kappa == 690.0
    assert est.xi == 730.0
    assert est.k == 1.0
    assert est.zero_consequent_init is True


def test_adptsk_classifier_paper_strict_rejects_conflicting_hyperparameters() -> None:
    with pytest.raises(ValueError, match="paper_strict requires n_mfs=3"):
        ADPTSKClassifier(paper_strict=True, n_mfs=4)
    with pytest.raises(ValueError, match="paper_strict requires mf_init='grid'"):
        ADPTSKClassifier(paper_strict=True, mf_init="kmeans")
    with pytest.raises(ValueError, match=r"paper_strict requires sigma_scale=1\.0"):
        ADPTSKClassifier(paper_strict=True, sigma_scale=0.8)
    with pytest.raises(ValueError, match="paper_strict requires rule_base='coco'"):
        ADPTSKClassifier(paper_strict=True, rule_base="cartesian")
    with pytest.raises(ValueError, match=r"paper_strict requires kappa=690\.0"):
        ADPTSKClassifier(paper_strict=True, kappa=700.0)
    with pytest.raises(ValueError, match=r"paper_strict requires xi=730\.0"):
        ADPTSKClassifier(paper_strict=True, xi=740.0)
    with pytest.raises(ValueError, match=r"paper_strict requires k=1\.0"):
        ADPTSKClassifier(paper_strict=True, k=0.8)
    with pytest.raises(ValueError, match="paper_strict requires zero_consequent_init=True"):
        ADPTSKClassifier(paper_strict=True, zero_consequent_init=False)


def test_adptsk_classifier_paper_strict_requires_inputs_in_unit_interval() -> None:
    x, y = _make_dataset(40)
    est = ADPTSKClassifier(paper_strict=True, epochs=1, random_state=0)
    with pytest.raises(ValueError, match=r"paper_strict requires x to be linearly normalized to \[0,1\]"):
        est.fit(x, y)


def test_adptsk_classifier_non_strict_accepts_inputs_outside_unit_interval() -> None:
    x, y = _make_dataset(40)
    est = ADPTSKClassifier(paper_strict=False, epochs=1, random_state=0)
    est.fit(x, y)
    assert est.model_ is not None


def test_adptsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = ADPTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_adptsk_regressor_default_estimator_uses_paper_rule_base_and_rule_count() -> None:
    x, y = _make_regression_dataset(80)
    est = ADPTSKRegressor(epochs=1, random_state=7)
    est.fit(x, y)
    assert est.rule_base_ == "coco"
    assert est.model_.n_rules == 3


def test_adptsk_regressor_default_estimator_uses_paper_centers_and_sigma() -> None:
    x, y = _make_regression_dataset(80)
    est = ADPTSKRegressor(epochs=1, random_state=7)
    est.fit(x, y)

    mfs = est.model_.input_mfs["x1"]
    assert len(mfs) == 3
    assert isinstance(mfs[0], GaussianPiMF)
    means = [float(cast(Tensor, cast(GaussianPiMF, mf).mean).detach().item()) for mf in mfs]
    sigmas = [float(cast(GaussianPiMF, mf).sigma.detach().item()) for mf in mfs]
    assert means == pytest.approx([0.0, 0.5, 1.0], abs=1e-8)
    assert sigmas == pytest.approx([1.0, 1.0, 1.0], abs=1e-8)


def test_adptsk_regressor_default_batch_size_policy() -> None:
    est = ADPTSKRegressor()
    assert est._resolve_default_batch_size(499) is None
    assert est._resolve_default_batch_size(500) == 100


# --- ADMTSK Estimator Tests ---


def test_admtsk_classifier_estimator_uses_composite_gmf() -> None:
    x, y = _make_dataset(80)
    est = ADMTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)
    assert all(isinstance(mf, GaussianPiMF) for mfs in est.model_.input_mfs.values() for mf in mfs)


def test_admtsk_classifier_default_estimator_uses_paper_rule_base_and_rule_count() -> None:
    x, y = _make_dataset(60)
    est = ADMTSKClassifier(epochs=1, random_state=7)
    est.fit(x, y)
    assert est.rule_base_ == "coco"
    assert est.model_.n_rules == 3


def test_admtsk_classifier_default_estimator_uses_paper_centers_and_sigma() -> None:
    x, y = _make_dataset(60)
    est = ADMTSKClassifier(epochs=1, random_state=7)
    est.fit(x, y)

    mfs = est.model_.input_mfs["x1"]
    assert len(mfs) == 3
    assert isinstance(mfs[0], GaussianPiMF)
    means = [float(cast(Tensor, cast(GaussianPiMF, mf).mean).detach().item()) for mf in mfs]
    sigmas = [float(cast(GaussianPiMF, mf).sigma.detach().item()) for mf in mfs]
    assert means == pytest.approx([0.0, 0.5, 1.0], abs=1e-8)
    assert sigmas == pytest.approx([1.0, 1.0, 1.0], abs=1e-8)


def test_admtsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = ADMTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_admtsk_regressor_default_estimator_uses_paper_centers_and_sigma() -> None:
    x, y = _make_regression_dataset(80)
    est = ADMTSKRegressor(epochs=1, random_state=7)
    est.fit(x, y)

    mfs = est.model_.input_mfs["x1"]
    assert len(mfs) == 3
    assert isinstance(mfs[0], GaussianPiMF)
    means = [float(cast(Tensor, cast(GaussianPiMF, mf).mean).detach().item()) for mf in mfs]
    sigmas = [float(cast(GaussianPiMF, mf).sigma.detach().item()) for mf in mfs]
    assert means == pytest.approx([0.0, 0.5, 1.0], abs=1e-8)
    assert sigmas == pytest.approx([1.0, 1.0, 1.0], abs=1e-8)


def test_admtsk_classifier_paper_strict_uses_paper_protocol_defaults() -> None:
    est = ADMTSKClassifier(paper_strict=True)
    assert est.n_mfs == 3
    assert est.mf_init == "grid"
    assert est.sigma_scale == 1.0
    assert est.rule_base == "coco"
    assert est.adaptive is True
    assert est.lambda_ == 1.0
    assert est.lower_bound == pytest.approx(1.0 / math.e, abs=1e-12)
    assert est.k == 10.0
    assert est.zero_consequent_init is True


def test_admtsk_classifier_paper_strict_rejects_conflicting_hyperparameters() -> None:
    with pytest.raises(ValueError, match="paper_strict requires n_mfs=3"):
        ADMTSKClassifier(paper_strict=True, n_mfs=5)
    with pytest.raises(ValueError, match="paper_strict requires mf_init='grid'"):
        ADMTSKClassifier(paper_strict=True, mf_init="kmeans")
    with pytest.raises(ValueError, match=r"paper_strict requires sigma_scale=1\.0"):
        ADMTSKClassifier(paper_strict=True, sigma_scale=2.0)
    with pytest.raises(ValueError, match="paper_strict requires rule_base='coco'"):
        ADMTSKClassifier(paper_strict=True, rule_base="cartesian")
    with pytest.raises(ValueError, match="paper_strict requires adaptive=True"):
        ADMTSKClassifier(paper_strict=True, adaptive=False)
    with pytest.raises(ValueError, match=r"paper_strict requires lambda_=1\.0"):
        ADMTSKClassifier(paper_strict=True, lambda_=2.0)
    with pytest.raises(ValueError, match="paper_strict requires lower_bound=1/e"):
        ADMTSKClassifier(paper_strict=True, lower_bound=0.3)
    with pytest.raises(ValueError, match=r"paper_strict requires k=10\.0"):
        ADMTSKClassifier(paper_strict=True, k=5.0)
    with pytest.raises(ValueError, match="paper_strict requires zero_consequent_init=True"):
        ADMTSKClassifier(paper_strict=True, zero_consequent_init=False)


def test_admtsk_classifier_paper_strict_fit_does_not_auto_split_data() -> None:
    x, y = _make_dataset(40)
    est = ADMTSKClassifier(paper_strict=True, epochs=1, random_state=0)
    est.fit(x, y)
    assert len(est.history_["val"]) == 0


def test_paper_strict_input_range_empty_arrays_adaptive() -> None:
    empty = np.array([])
    _validate_adptsk_paper_strict_input_range(empty)
    _validate_adatsk_paper_strict_input_range(empty)


def test_fit_with_strict_and_validation_data_adaptive() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, (40, 2)).astype(np.float32)
    y = rng.choice([0, 1], size=40).astype(np.int64)
    x_val = rng.uniform(0.0, 1.0, (10, 2)).astype(np.float32)
    y_val = rng.choice([0, 1], size=10).astype(np.int64)

    clf_adp = ADPTSKClassifier(paper_strict=True, epochs=1)
    clf_adp.fit(x, y, x_val=x_val, y_val=y_val)

    clf_ada = ADATSKClassifier(paper_strict=True, epochs=1)
    clf_ada.fit(x, y, x_val=x_val, y_val=y_val)
