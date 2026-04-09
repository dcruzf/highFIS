from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from highfis.estimators import (
    HTSKClassifierEstimator,
    InputConfig,
    _build_gaussian_input_mfs,
    _build_kmeans_input_mfs,
)


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return x, y


def test_build_gaussian_input_mfs_uses_input_configs() -> None:
    x, _ = _make_dataset(20)
    configs = [
        InputConfig(name="x1", n_mfs=2),
        InputConfig(name="x2", n_mfs=3),
        InputConfig(name="x3", n_mfs=4),
    ]

    input_mfs = _build_gaussian_input_mfs(x, configs)

    assert list(input_mfs.keys()) == ["x1", "x2", "x3"]
    assert len(input_mfs["x1"]) == 2
    assert len(input_mfs["x2"]) == 3
    assert len(input_mfs["x3"]) == 4


def test_build_gaussian_input_mfs_validates_n_mfs() -> None:
    x, _ = _make_dataset(20)
    with pytest.raises(ValueError, match="n_mfs"):
        _build_gaussian_input_mfs(x, [InputConfig(name="x1", n_mfs=0)])


def test_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = HTSKClassifierEstimator(
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


def test_estimator_grid_init_fit_predict() -> None:
    x, y = _make_dataset(80)
    est = HTSKClassifierEstimator(
        n_mfs=2,
        mf_init="grid",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    proba = est.predict_proba(x)
    pred = est.predict(x)

    assert proba.shape == (x.shape[0], 2)
    assert pred.shape == (x.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_estimator_predict_proba_requires_fit() -> None:
    x, _ = _make_dataset(10)
    est = HTSKClassifierEstimator(n_mfs=2, epochs=1, batch_size=16)
    with pytest.raises(NotFittedError):
        est.predict_proba(x)


def test_estimator_validates_input_config_length() -> None:
    x, y = _make_dataset(20)
    est = HTSKClassifierEstimator(input_configs=[InputConfig(name="x1", n_mfs=2)], batch_size=16)
    with pytest.raises(ValueError, match="input_configs length"):
        est.fit(x, y)


# ---------------------------------------------------------------------------
# _build_kmeans_input_mfs unit tests
# ---------------------------------------------------------------------------


def test_build_kmeans_input_mfs_shape() -> None:
    x, _ = _make_dataset(60)
    feature_names = ["a", "b", "c"]
    n_clusters = 4

    input_mfs = _build_kmeans_input_mfs(
        x,
        n_clusters=n_clusters,
        sigma_scale=1.0,
        feature_names=feature_names,
        random_state=0,
    )

    assert list(input_mfs.keys()) == feature_names
    for name in feature_names:
        assert len(input_mfs[name]) == n_clusters


def test_build_kmeans_input_mfs_sigma_positive() -> None:
    x, _ = _make_dataset(60)
    feature_names = ["x1", "x2", "x3"]

    input_mfs = _build_kmeans_input_mfs(
        x, n_clusters=3, sigma_scale=1.0, feature_names=feature_names, random_state=42
    )

    from highfis.memberships import GaussianMF

    for mfs in input_mfs.values():
        for mf in mfs:
            assert isinstance(mf, GaussianMF)
            sigma_val = float(mf.sigma.detach())
            assert sigma_val > 0, f"sigma must be positive, got {sigma_val}"


def test_build_kmeans_sigma_scale_applied() -> None:
    x, _ = _make_dataset(60)
    feature_names = ["x1", "x2", "x3"]

    mfs_1 = _build_kmeans_input_mfs(x, 3, sigma_scale=1.0, feature_names=feature_names, random_state=0)
    mfs_2 = _build_kmeans_input_mfs(x, 3, sigma_scale=5.0, feature_names=feature_names, random_state=0)

    from highfis.memberships import GaussianMF

    for name in feature_names:
        for m1, m2 in zip(mfs_1[name], mfs_2[name]):
            assert isinstance(m1, GaussianMF)
            assert isinstance(m2, GaussianMF)
            # sigma_scale=5 must produce >= sigma_scale=1
            assert float(m2.sigma.detach()) >= float(m1.sigma.detach()) - 1e-6


def test_estimator_invalid_mf_init_raises() -> None:
    x, y = _make_dataset(20)
    est = HTSKClassifierEstimator(n_mfs=2, mf_init="random", epochs=1, batch_size=16)
    with pytest.raises(ValueError, match="mf_init"):
        est.fit(x, y)


def test_estimator_kmeans_default_rule_base_is_coco() -> None:
    """When mf_init='kmeans' and rule_base is not set, model uses 'coco' rule base."""
    x, y = _make_dataset(60)
    est = HTSKClassifierEstimator(n_mfs=3, mf_init="kmeans", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)
    # With coco + 3 clusters and 3 features → 3 rules
    assert est.model_.n_rules == 3  # type: ignore[attr-defined]


def test_estimator_early_stopping_with_validation_data() -> None:
    """Estimator stops early when validation_data is provided."""
    x, y = _make_dataset(80)
    x_train, x_val = x[:60], x[60:]
    y_train, y_val = y[:60], y[60:]

    est = HTSKClassifierEstimator(
        n_mfs=2,
        mf_init="kmeans",
        epochs=1000,
        learning_rate=5e-2,
        random_state=7,
        patience=3,
        validation_data=(x_val, y_val),
    )
    est.fit(x_train, y_train)

    assert "val" in est.history_
    assert len(est.history_["val"]) > 0
    assert est.history_["stopped_epoch"] < 1000


def test_estimator_no_val_runs_full_epochs() -> None:
    """Without validation_data, training runs for the full epoch count."""
    x, y = _make_dataset(60)
    est = HTSKClassifierEstimator(
        n_mfs=2, mf_init="kmeans", epochs=10, random_state=7, batch_size=16,
    )
    est.fit(x, y)

    assert est.history_["stopped_epoch"] == 10
    assert len(est.history_["val"]) == 0


def test_estimator_sigma_scale_auto() -> None:
    """sigma_scale='auto' uses h=sqrt(D) where D is the number of features."""
    x, y = _make_dataset(60)
    est = HTSKClassifierEstimator(
        n_mfs=3, mf_init="kmeans", sigma_scale="auto",
        epochs=2, random_state=0, batch_size=16,
    )
    est.fit(x, y)

    # With 3 features, auto → h=sqrt(3) ≈ 1.73
    assert est.model_ is not None
    proba = est.predict_proba(x)
    assert proba.shape == (x.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# _build_gaussian_input_mfs edge cases
# ---------------------------------------------------------------------------


def test_build_gaussian_input_mfs_constant_column() -> None:
    """All values identical → rmax == rmin → rmax += 1e-3 branch (line 46)."""
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(20), rng.normal(size=20)]).astype(np.float64)
    configs = [InputConfig(name="x1", n_mfs=2), InputConfig(name="x2", n_mfs=2)]
    mfs = _build_gaussian_input_mfs(x, configs)
    assert len(mfs["x1"]) == 2


def test_build_gaussian_input_mfs_single_mf() -> None:
    """n_mfs==1 triggers width=range branch (line 50)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 1)).astype(np.float64)
    configs = [InputConfig(name="x1", n_mfs=1)]
    mfs = _build_gaussian_input_mfs(x, configs)
    assert len(mfs["x1"]) == 1
    assert float(mfs["x1"][0].sigma.detach()) > 0


# ---------------------------------------------------------------------------
# _build_kmeans_input_mfs zero-sigma fallback
# ---------------------------------------------------------------------------


def test_build_kmeans_zero_sigma_fallback() -> None:
    """Constant-value cluster → std=0 → gap-fallback branch (lines 95-96)."""
    x = np.vstack([
        np.zeros((10, 2), dtype=np.float64),
        np.ones((10, 2), dtype=np.float64),
    ])
    mfs = _build_kmeans_input_mfs(
        x, n_clusters=2, sigma_scale=1.0, feature_names=["x1", "x2"], random_state=0,
    )
    for name in ["x1", "x2"]:
        for mf in mfs[name]:
            assert float(mf.sigma.detach()) > 0


# ---------------------------------------------------------------------------
# _resolve_input_configs and _resolve_feature_names happy paths
# ---------------------------------------------------------------------------


def test_estimator_fit_with_input_configs_grid_resolve_config() -> None:
    """input_configs set + grid init → _resolve_input_configs happy path (lines 171-175)."""
    x, y = _make_dataset(60)
    configs = [InputConfig(name=f"f{i}", n_mfs=2) for i in range(3)]
    est = HTSKClassifierEstimator(
        input_configs=configs, mf_init="grid", epochs=2, random_state=0, batch_size=16,
    )
    est.fit(x, y)
    assert list(est.feature_names_in_) == ["f0", "f1", "f2"]


def test_estimator_fit_with_input_configs_kmeans_resolve_names() -> None:
    """input_configs set + kmeans init → _resolve_feature_names happy path (line 184)."""
    x, y = _make_dataset(60)
    configs = [InputConfig(name=f"g{i}", n_mfs=3) for i in range(3)]
    est = HTSKClassifierEstimator(
        input_configs=configs, mf_init="kmeans", epochs=2, random_state=0, batch_size=16,
    )
    est.fit(x, y)
    assert list(est.feature_names_in_) == ["g0", "g1", "g2"]


# ---------------------------------------------------------------------------
# predict_proba feature-count validation
# ---------------------------------------------------------------------------


def test_estimator_predict_proba_wrong_feature_count() -> None:
    """predict_proba with wrong number of features raises ValueError (line 271)."""
    x, y = _make_dataset(40)
    est = HTSKClassifierEstimator(n_mfs=2, epochs=2, batch_size=16, random_state=0)
    est.fit(x, y)
    with pytest.raises(ValueError, match="expected"):
        est.predict_proba(x[:, :2])
