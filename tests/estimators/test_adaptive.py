from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn

from highfis import (
    ADATSKClassifier,
    ADATSKRegressor,
    ADMTSKClassifier,
    ADMTSKRegressor,
    ADPTSKClassifier,
    ADPTSKRegressor,
)
from highfis.estimators import InputConfig
from highfis.estimators._adaptive import _set_sigma_to_one_and_freeze, _wrap_adatsk_gaussian_input_mfs
from highfis.memberships import ADATSKGaussianMF, CompositeGaussianMF, GaussianMF, GaussianPiMF


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return (x, y)


def _make_regression_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(456)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2]).astype(np.float32)
    return (x, y)


def test_adatsk_classifier_pre_hook_sets_sigma_one_and_freezes_sigma() -> None:
    x = np.random.default_rng(0).normal(size=(12, 3)).astype(np.float64)
    y = np.random.default_rng(1).integers(0, 2, size=(12,), dtype=np.int64)
    clf = ADATSKClassifier(epochs=1, high_dim_threshold=10000)
    clf.fit(x, y)
    for mf_list in clf.model_.membership_layer.input_mfs.values():
        for module in cast(nn.ModuleList, mf_list):
            mf = cast(GaussianMF, module)
            assert abs(float(mf.sigma.detach().item()) - 1.0) < 0.001
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
    assert torch.allclose(centers, expected, atol=1e-06)


def test_set_sigma_to_one_and_freeze_ignores_non_gaussian_mf() -> None:
    mf = CompositeGaussianMF(mean=0.0, sigma=0.7, eps=0.0001)
    before = float(mf.sigma.detach().item())
    assert mf.raw_sigma.requires_grad is True
    _set_sigma_to_one_and_freeze(mf)
    after = float(mf.sigma.detach().item())
    assert abs(after - before) < 1e-06
    assert mf.raw_sigma.requires_grad is True


def test_wrap_adatsk_gaussian_input_mfs_preserves_non_gaussian_modules() -> None:
    cg = CompositeGaussianMF(mean=0.0, sigma=0.9, eps=0.0001)
    g = GaussianMF(mean=1.0, sigma=1.1, eps=0.0001)
    wrapped = _wrap_adatsk_gaussian_input_mfs({"x1": [g, cg]})
    assert isinstance(wrapped["x1"][0], ADATSKGaussianMF)
    assert wrapped["x1"][1] is cg


def test_adatsk_classifier_resolve_input_configs_keeps_user_configs() -> None:
    configs = [InputConfig(name="x1", n_mfs=3, overlap=0.5, margin=0.2)]
    clf = ADATSKClassifier(input_configs=configs)
    x = np.array([[0.0], [1.0]], dtype=np.float64)
    resolved = clf._resolve_input_configs(x)
    assert resolved[0].margin == 0.2


def test_composite_gaussian_mf_lower_bound() -> None:
    mf = CompositeGaussianMF(mean=0.0, sigma=1.0, eps=0.05)
    x = torch.tensor([-5.0, 0.0, 5.0])
    values = mf(x)
    assert torch.all(values >= 0.05)
    assert torch.all(values <= 1.0)


def test_adatsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = ADATSKClassifier(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    proba = est.predict_proba(x)
    pred = est.predict(x)
    score = est.score(x, y)
    assert proba.shape == (x.shape[0], 2)
    assert pred.shape == (x.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-06)
    assert 0.0 <= score <= 1.0


def test_adatsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = ADATSKRegressor(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_adptsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = ADPTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    proba = est.predict_proba(x)
    pred = est.predict(x)
    score = est.score(x, y)
    assert proba.shape == (x.shape[0], 2)
    assert pred.shape == (x.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-06)
    assert 0.0 <= score <= 1.0


def test_adptsk_default_batch_size_policy() -> None:
    est = ADPTSKClassifier()
    assert est._resolve_default_batch_size(499) is None
    assert est._resolve_default_batch_size(500) == 100


def test_adptsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = ADPTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_adptsk_regressor_default_batch_size_policy() -> None:
    est = ADPTSKRegressor()
    assert est._resolve_default_batch_size(499) is None
    assert est._resolve_default_batch_size(500) == 100


def test_admtsk_classifier_estimator_uses_composite_gmf() -> None:
    x, y = _make_dataset(80)
    est = ADMTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=3, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    assert all(isinstance(mf, GaussianPiMF) for mfs in est.model_.input_mfs.values() for mf in mfs)


def test_admtsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = ADMTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=3, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_adptsk_default_profile() -> None:
    x, y = _make_dataset(20)
    # Add constant column to trigger x_max <= x_min (line 42)
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    clf = ADPTSKClassifier(epochs=1)
    clf.fit(x, y)
    # Check that default grid init with 3 mfs works
    assert clf.model_ is not None

    x_reg, y_reg = _make_regression_dataset(20)
    x_reg = np.hstack([x_reg, np.ones((x_reg.shape[0], 1))])
    reg = ADPTSKRegressor(epochs=1)
    reg.fit(x_reg, y_reg)
    assert reg.model_ is not None


def test_adptsk_nan_guards() -> None:
    x, y = _make_dataset(10)
    clf = ADPTSKClassifier(epochs=1)
    clf.fit(x, y)

    # Mock model predict_proba to return NaNs
    clf.model_.predict_proba = lambda *args, **kwargs: torch.full((10, 2), float("nan"))  # type: ignore
    proba = clf.predict_proba(x)
    assert not np.any(np.isnan(proba))
    assert np.allclose(proba, 0.5)

    x_reg, y_reg = _make_regression_dataset(10)
    reg = ADPTSKRegressor(epochs=1)
    reg.fit(x_reg, y_reg)

    reg.model_.predict = lambda *args, **kwargs: torch.full((10, 1), float("nan"))  # type: ignore
    pred = reg.predict(x_reg)
    assert not np.any(np.isnan(pred))
    assert np.allclose(pred, 0.0)


def test_adptsk_low_dimensional_no_nan() -> None:
    """Integration test: ADPTSK should not diverge to NaN on low-dimensional data after fix.

    This replicates the scenario from BUG_REPORT_ADPTSK_NaN.md: low-dimensional
    data (D=2) with ADPTSK typically triggered NaN on first gradient step due to
    a division-by-zero pole in ADPSoftminRuleLayer. After the fix, this should pass.
    """
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=60, noise=0.1, factor=0.5, random_state=42)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # Normalize to [0, 1]

    # Use settings that would trigger NaN in the old code
    clf = ADPTSKClassifier(
        n_mfs=5,
        mf_init="kmeans",
        rule_base="coco",
        epochs=5,
        learning_rate=0.05,
        random_state=42,
        verbose=False,
    )
    clf.fit(X[:40], y[:40])

    # After training, forward pass through model should not contain NaN
    X_test_tensor = torch.tensor(X[40:], dtype=torch.get_default_dtype())
    with torch.no_grad():
        antecedent_output = clf.model_.forward_antecedents(X_test_tensor)
    assert torch.all(torch.isfinite(antecedent_output)), "forward_antecedents contains NaN/inf after training"

    # predict and predict_proba should work without falling back to NaN guards
    proba = clf.predict_proba(X[40:])
    pred = clf.predict(X[40:])
    assert proba.shape == (20, 2)
    assert pred.shape == (20,)
    assert not np.any(np.isnan(proba))
    assert not np.any(np.isnan(pred))
