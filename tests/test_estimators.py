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
    est = HTSKClassifierEstimator(n_mfs=2, epochs=1)
    with pytest.raises(NotFittedError):
        est.predict_proba(x)


def test_estimator_validates_input_config_length() -> None:
    x, y = _make_dataset(20)
    est = HTSKClassifierEstimator(input_configs=[InputConfig(name="x1", n_mfs=2)])
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
    est = HTSKClassifierEstimator(n_mfs=2, mf_init="random", epochs=1)
    with pytest.raises(ValueError, match="mf_init"):
        est.fit(x, y)


def test_estimator_kmeans_default_rule_base_is_coco() -> None:
    """When mf_init='kmeans' and rule_base is not set, model uses 'coco' rule base."""
    x, y = _make_dataset(60)
    est = HTSKClassifierEstimator(n_mfs=3, mf_init="kmeans", epochs=2, random_state=0)
    est.fit(x, y)
    # With coco + 3 clusters and 3 features → 3 rules
    assert est.model_.n_rules == 3  # type: ignore[attr-defined]
