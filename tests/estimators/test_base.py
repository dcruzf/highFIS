from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch

from highfis import HTSKClassifier
from highfis.clustering import FuzzyCMeans, KMeans
from highfis.estimators import InputConfig
from highfis.estimators._base import (
    _build_fuzzy_c_means_input_mfs,
    _build_gaussian_input_mfs,
    _build_kmeans_input_mfs,
    _build_pfrb_input_mfs,
    _mann_whitney_p_value,
    _normalize_importance,
    _rankdata,
    _resolve_mhtsk_scale_parameters,
    _select_rule_indices,
)
from highfis.memberships import GaussianMF


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return (x, y)


def test_build_gaussian_input_mfs_uses_input_configs() -> None:
    x, _ = _make_dataset(20)
    configs = [InputConfig(name="x1", n_mfs=2), InputConfig(name="x2", n_mfs=3), InputConfig(name="x3", n_mfs=4)]
    input_mfs = _build_gaussian_input_mfs(x, configs)
    assert list(input_mfs.keys()) == ["x1", "x2", "x3"]
    assert len(input_mfs["x1"]) == 2
    assert len(input_mfs["x2"]) == 3
    assert len(input_mfs["x3"]) == 4


def test_build_gaussian_input_mfs_validates_n_mfs() -> None:
    x, _ = _make_dataset(20)
    with pytest.raises(ValueError, match="n_mfs"):
        _build_gaussian_input_mfs(x, [InputConfig(name="x1", n_mfs=0)])


def test_build_fuzzy_c_means_input_mfs_handles_zero_variance() -> None:
    class DummyFuzzyCMeans:
        def __init__(self) -> None:
            self.m = 2.0
            self.n_clusters = 2
            self.random_state = 0
            self.cluster_centers_ = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
            self.membership_ = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)

        def fit(self, x: np.ndarray) -> DummyFuzzyCMeans:
            return self

    x = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    feature_names = ["x1", "x2"]
    input_mfs = _build_fuzzy_c_means_input_mfs(
        x, cast(FuzzyCMeans, DummyFuzzyCMeans()), sigma_scale=1.0, feature_names=feature_names, random_state=0
    )
    assert list(input_mfs.keys()) == feature_names
    assert len(input_mfs["x1"]) == 2
    assert all(isinstance(mf, GaussianMF) for mf in input_mfs["x1"])
    assert input_mfs["x1"][0].sigma > 0.0


def test_build_pfrb_input_mfs_limits_rules_and_returns_gaussian_mfs() -> None:
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    feature_names = ["x1", "x2", "x3"]
    input_mfs = _build_pfrb_input_mfs(x, feature_names, max_rules=2, sigma_scale=1.0, random_state=0)
    assert list(input_mfs.keys()) == feature_names
    assert len(input_mfs["x1"]) == 2
    assert len(input_mfs["x2"]) == 2
    assert len(input_mfs["x3"]) == 2
    for mfs in input_mfs.values():
        for mf in mfs:
            assert isinstance(mf, GaussianMF)
            assert float(mf.sigma.detach()) > 0.0


def test_build_pfrb_input_mfs_max_rules_none_uses_all_samples() -> None:
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    feature_names = ["x1", "x2", "x3"]
    input_mfs = _build_pfrb_input_mfs(x, feature_names, max_rules=None, sigma_scale=1.0, random_state=0)
    assert len(input_mfs["x1"]) == 4
    assert len(input_mfs["x2"]) == 4
    assert len(input_mfs["x3"]) == 4


def test_build_kmeans_input_mfs_shape() -> None:
    x, _ = _make_dataset(60)
    feature_names = ["a", "b", "c"]
    n_clusters = 4
    input_mfs = _build_kmeans_input_mfs(
        x, KMeans(n_clusters=n_clusters, random_state=0), sigma_scale=1.0, feature_names=feature_names, random_state=0
    )
    assert list(input_mfs.keys()) == feature_names
    for name in feature_names:
        assert len(input_mfs[name]) == n_clusters


def test_build_kmeans_input_mfs_raises_when_cluster_centers_missing() -> None:
    class DummyKMeans:
        def __init__(self):
            self.cluster_centers_ = None
            self.labels_ = np.array([], dtype=int)
            self.n_clusters = 2
            self.random_state = 0

        def fit(self, x: np.ndarray):
            return self

    with pytest.raises(RuntimeError, match="KMeans did not compute cluster centers"):
        _build_kmeans_input_mfs(
            np.zeros((2, 2), dtype=np.float64),
            cast(KMeans, DummyKMeans()),
            sigma_scale=1.0,
            feature_names=["x1", "x2"],
            random_state=0,
        )


def test_build_kmeans_input_mfs_sigma_positive() -> None:
    x, _ = _make_dataset(60)
    feature_names = ["x1", "x2", "x3"]
    input_mfs = _build_kmeans_input_mfs(
        x, KMeans(n_clusters=3, random_state=42), sigma_scale=1.0, feature_names=feature_names, random_state=42
    )
    for mfs in input_mfs.values():
        for mf in mfs:
            assert isinstance(mf, GaussianMF)
            sigma_val = float(mf.sigma.detach())
            assert sigma_val > 0, f"sigma must be positive, got {sigma_val}"


def test_build_kmeans_sigma_scale_applied() -> None:
    x, _ = _make_dataset(60)
    feature_names = ["x1", "x2", "x3"]
    mfs_1 = _build_kmeans_input_mfs(
        x, KMeans(n_clusters=3, random_state=0), sigma_scale=1.0, feature_names=feature_names, random_state=0
    )
    mfs_2 = _build_kmeans_input_mfs(
        x, KMeans(n_clusters=3, random_state=0), sigma_scale=5.0, feature_names=feature_names, random_state=0
    )
    for name in feature_names:
        for m1, m2 in zip(mfs_1[name], mfs_2[name], strict=False):
            assert isinstance(m1, GaussianMF)
            assert isinstance(m2, GaussianMF)
            assert float(m2.sigma.detach()) >= float(m1.sigma.detach()) - 1e-06


def test_build_fcm_input_mfs_zero_weight_fallback() -> None:
    x = np.zeros((3, 2), dtype=np.float64)
    feature_names = ["x1", "x2"]
    input_mfs = _build_fuzzy_c_means_input_mfs(
        x, FuzzyCMeans(n_clusters=2, random_state=0), sigma_scale=1.0, feature_names=feature_names, random_state=0
    )
    assert list(input_mfs.keys()) == feature_names
    for mfs in input_mfs.values():
        assert len(mfs) == 2
        for mf in mfs:
            assert isinstance(mf, GaussianMF)
            assert float(mf.sigma.detach()) > 0.0


def test_build_fcm_input_mfs_raises_when_fcm_does_not_converge() -> None:
    class DummyFuzzyCMeans:
        def __init__(self):
            self.cluster_centers_ = None
            self.membership_ = None
            self.m = 2.0
            self.n_clusters = 2
            self.random_state = 0

        def fit(self, x: np.ndarray):
            return self

    with pytest.raises(RuntimeError, match="FuzzyCMeans did not converge"):
        _build_fuzzy_c_means_input_mfs(
            np.zeros((2, 2), dtype=np.float64),
            cast(FuzzyCMeans, DummyFuzzyCMeans()),
            sigma_scale=1.0,
            feature_names=["x1", "x2"],
            random_state=0,
        )


def test_build_kmeans_input_mfs_works_with_numpy_cluster_centers() -> None:
    class DummyKMeans:
        def __init__(self):
            self.cluster_centers_ = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
            self.labels_ = np.array([0, 1], dtype=int)
            self.n_clusters = 2
            self.random_state = 0

        def fit(self, x: np.ndarray):
            return self

    x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    feature_names = ["x1", "x2"]
    input_mfs = _build_kmeans_input_mfs(
        x, cast(KMeans, DummyKMeans()), sigma_scale=1.0, feature_names=feature_names, random_state=0
    )
    assert list(input_mfs.keys()) == feature_names
    assert len(input_mfs["x1"]) == 2


def test_rankdata_handles_ties() -> None:
    values = np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float64)
    ranks = _rankdata(values)
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_mann_whitney_p_value_with_empty_group() -> None:
    assert _mann_whitney_p_value(np.array([], dtype=np.float64), np.array([1.0])) == 1.0


def test_select_rule_indices_edge_cases() -> None:
    assert _select_rule_indices(torch.tensor([], dtype=torch.float32), 0.5) == []
    assert _select_rule_indices(torch.tensor([1.0, 2.0]), 0.0) == []
    assert _select_rule_indices(torch.tensor([1.0, 2.0]), 1.0) == [0, 1]
    zero_sum = torch.tensor([0.0, 0.0], dtype=torch.float32)
    assert _select_rule_indices(zero_sum, 0.5) == [0]
    with pytest.raises(ValueError, match="crcr must be between 0 and 1"):
        _select_rule_indices(torch.tensor([1.0, 2.0]), -0.1)


def test_normalize_importance_returns_uniform_distribution_for_zero_total() -> None:
    values = torch.zeros(3, dtype=torch.float32)
    normalized = _normalize_importance(values)
    assert normalized.shape == (3,)
    assert np.allclose(normalized, np.array([1.0 / 3.0] * 3))


def test_membership_functions_initialization_caching() -> None:
    from highfis.estimators._base import _MF_INITIALIZATION_CACHE

    x, _ = _make_dataset(40)
    _MF_INITIALIZATION_CACHE.clear()
    est1 = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    mfs1, names1, rb1 = est1._build_input_mfs(x)
    assert len(_MF_INITIALIZATION_CACHE) == 1
    est2 = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    mfs2, names2, rb2 = est2._build_input_mfs(x)
    assert len(_MF_INITIALIZATION_CACHE) == 1
    assert list(mfs1.keys()) == list(mfs2.keys())
    assert names1 == names2
    assert rb1 == rb2
    for name in mfs1:
        assert mfs1[name] is not mfs2[name]
        for mf1, mf2 in zip(mfs1[name], mfs2[name], strict=True):
            assert mf1 is not mf2
            assert mf1.mean is not mf2.mean
            val1 = float(cast(Any, mf1.mean).detach().cpu().numpy())
            val2 = float(cast(Any, mf2.mean).detach().cpu().numpy())
            assert val1 == val2
    est3 = HTSKClassifier(n_mfs=3, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est3._build_input_mfs(x)
    assert len(_MF_INITIALIZATION_CACHE) == 2


def test_membership_functions_initialization_caching_eviction() -> None:
    from highfis.estimators._base import _MF_INITIALIZATION_CACHE

    x, _ = _make_dataset(40)
    _MF_INITIALIZATION_CACHE.clear()
    for rs in range(1, 129):
        est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=rs, batch_size=16)
        est._build_input_mfs(x)
    assert len(_MF_INITIALIZATION_CACHE) == 128
    keys_before = list(_MF_INITIALIZATION_CACHE.keys())
    est_new = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=129, batch_size=16)
    est_new._build_input_mfs(x)
    assert len(_MF_INITIALIZATION_CACHE) == 128
    assert keys_before[0] not in _MF_INITIALIZATION_CACHE
    assert keys_before[1] in _MF_INITIALIZATION_CACHE


def test_get_mf_cache_key_edge_cases() -> None:
    from highfis.estimators._base import _get_mf_cache_key

    x, _ = _make_dataset(20)
    key1 = _get_mf_cache_key(x, None, 2, 1.0, 42, None, None)
    assert key1[1] is None
    clusterer = KMeans(n_clusters=2)
    key2 = _get_mf_cache_key(x, clusterer, 2, 1.0, 42, None, None)
    assert key2[1] == ("KMeans", 2, None)


def test_classifier_input_configs_mismatch() -> None:
    x, y = _make_dataset(20)
    configs = [InputConfig(name="x1", n_mfs=2)]
    est = HTSKClassifier(input_configs=configs, mf_init="grid")
    with pytest.raises(ValueError, match="input_configs length"):
        est.fit(x, y)


def test_regressor_input_configs_mismatch() -> None:
    from highfis import HTSKRegressor

    x, y = _make_dataset(20)
    configs = [InputConfig(name="x1", n_mfs=2)]
    est = HTSKRegressor(input_configs=configs, mf_init="grid")
    with pytest.raises(ValueError, match="input_configs length"):
        est.fit(x, y)
    matching_configs = [InputConfig(name=f"x{i}", n_mfs=2) for i in range(3)]
    est_matching = HTSKRegressor(input_configs=matching_configs, mf_init="grid", epochs=1)
    est_matching.fit(x, y)


def test_regressor_grid_pfrb() -> None:
    from highfis import HTSKRegressor

    x, y = _make_dataset(20)
    est = HTSKRegressor(mf_init="grid", rule_base="pfrb", pfrb_max_rules=2, sigma_scale=1.0)
    est.fit(x, y)
    assert est.rule_base_ == "coco"


def test_regressor_sigma_scale_auto() -> None:
    from highfis import HTSKRegressor

    x, y = _make_dataset(20)
    est = HTSKRegressor(sigma_scale="auto")
    est.fit(x, y)


def test_resolve_mhtsk_scale_parameters_large_features() -> None:
    # Test high feature count branch (n_features > 5000) for line 372
    h, n = _resolve_mhtsk_scale_parameters(
        n_features=6000,
        head_size=None,
        head_size_ratio=None,
        n_heads=None,
        fcr_target=0.85,
        h_value=None,
        sigma=10.0,
        xi=1.0,
    )
    assert h == 60  # round(6000 * 0.01)
    assert n > 0


def test_fit_read_only_inputs_classifier() -> None:
    # Test read-only array conversion for line 858
    x, y = _make_dataset(20)
    x = x.copy()
    x.setflags(write=False)
    est = HTSKClassifier(epochs=1)
    est.fit(x, y)


def test_fit_read_only_inputs_regressor() -> None:
    # Test read-only array conversion for line 1367
    from highfis import HTSKRegressor

    x, y = _make_dataset(20)
    x = x.copy()
    x.setflags(write=False)
    est = HTSKRegressor(epochs=1)
    est.fit(x, y)


def test_classifier_fit_too_few_samples() -> None:
    # Test minimum samples exception for line 975
    x, y = _make_dataset(1)  # only 1 sample
    est = HTSKClassifier(epochs=1)
    with pytest.raises(ValueError, match="requires at least 3 samples"):
        est.fit(x, y)


def test_classifier_fit_continuous_labels() -> None:
    # Test label type exception for line 980
    x, _ = _make_dataset(20)
    y_continuous = np.random.randn(20)
    est = HTSKClassifier(epochs=1)
    with pytest.raises(ValueError, match="Unknown label type: continuous"):
        est.fit(x, y_continuous)


def test_regressor_fit_too_few_samples() -> None:
    # Test minimum samples exception for line 1479
    from highfis import HTSKRegressor

    x, y = _make_dataset(1)  # only 1 sample
    est = HTSKRegressor(epochs=1)
    with pytest.raises(ValueError, match="requires at least 3 samples"):
        est.fit(x, y)
