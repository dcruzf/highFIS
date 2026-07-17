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
    _normalize_importance,
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


def test_normalize_importance_returns_uniform_distribution_for_zero_total() -> None:
    values = torch.zeros(3, dtype=torch.float32)
    normalized = _normalize_importance(values)
    assert normalized.shape == (3,)
    assert np.allclose(normalized, np.array([1.0 / 3.0] * 3))


def test_membership_functions_initialization_caching() -> None:
    from highfis import clear_mf_cache, mf_cache_info

    x, _ = _make_dataset(40)
    clear_mf_cache()
    est1 = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    mfs1, names1, rb1 = est1._build_input_mfs(x)
    assert mf_cache_info().currsize == 1
    est2 = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    mfs2, names2, rb2 = est2._build_input_mfs(x)
    assert mf_cache_info().currsize == 1
    assert mf_cache_info().hits == 1  # second build was a cache hit
    assert list(mfs1.keys()) == list(mfs2.keys())
    assert names1 == names2
    assert rb1 == rb2
    for name in mfs1:
        assert mfs1[name] is not mfs2[name]
        for mf1, mf2 in zip(mfs1[name], mfs2[name], strict=True):
            assert mf1 is not mf2
            assert mf1.mean is not mf2.mean
            val1 = float(mf1.mean.detach().cpu().numpy())
            val2 = float(mf2.mean.detach().cpu().numpy())
            assert val1 == val2
    est3 = HTSKClassifier(n_mfs=3, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est3._build_input_mfs(x)
    assert mf_cache_info().currsize == 2


def test_membership_functions_initialization_caching_eviction() -> None:
    from highfis import clear_mf_cache, mf_cache_info

    x, _ = _make_dataset(40)
    clear_mf_cache()
    for rs in range(1, 129):
        est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=rs, batch_size=16)
        est._build_input_mfs(x)
    assert mf_cache_info().currsize == 128
    est_new = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=129, batch_size=16)
    est_new._build_input_mfs(x)
    # Adding a 129th distinct key keeps the cache capped at maxsize (LRU eviction).
    assert mf_cache_info().currsize == 128
    assert mf_cache_info().maxsize == 128


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


def test_build_input_mfs_cached_thread_safety() -> None:
    import threading
    import time

    from highfis.estimators._base import _build_input_mfs_cached

    class DummyEstimator:
        def __init__(self) -> None:
            self.mf_init = "kmeans"
            self.n_mfs = 3
            self.sigma_scale = 1.0
            self.random_state = 42
            self.pfrb_max_rules = None
            self.input_configs = None

    estimator = DummyEstimator()
    x = np.random.randn(20, 2)

    def build_func(x_arr: np.ndarray) -> tuple[Any, list[str], str]:
        from highfis.memberships import GaussianMF

        time.sleep(0.01)
        mfs = {"x1": [GaussianMF(mean=0.0, sigma=1.0)], "x2": [GaussianMF(mean=1.0, sigma=1.0)]}
        return mfs, ["x1", "x2"], "coco"

    threads = []
    errors = []

    def worker() -> None:
        try:
            _mfs, names, rb = _build_input_mfs_cached(estimator, x, build_func)
            assert len(names) == 2
            assert rb == "coco"
        except Exception as e:
            errors.append(e)

    for _ in range(10):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0


def test_classifier_save_load_with_rule_layer_rules() -> None:
    """Test save/load for models with rule_layer.rules (lines 789-791)."""
    import tempfile

    x, y = _make_dataset(20)
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est.fit(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.ckpt"
        est.save(path)
        est_loaded = HTSKClassifier.load(path)
        assert est_loaded.model_ is not None
        assert est_loaded.classes_.shape == est.classes_.shape


def test_regressor_save_load_with_rule_layer_rules() -> None:
    """Test save/load for regressor models with rule_layer.rules (lines 992-994)."""
    import tempfile

    from highfis import HTSKRegressor

    x, y = _make_dataset(20)
    est = HTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est.fit(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.ckpt"
        est.save(path)
        est_loaded = HTSKRegressor.load(path)
        assert est_loaded.model_ is not None
        assert est_loaded.n_features_in_ == est.n_features_in_


def test_classifier_load_without_rules_support() -> None:
    """Test classifier load path when _build_model doesn't support rules (line 832)."""
    import inspect
    import tempfile
    from unittest.mock import patch

    x, y = _make_dataset(20)
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est.fit(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.ckpt"
        est.save(path)

        # Mock inspect.signature to simulate a _build_model without 'rules' parameter
        original_signature = inspect.signature

        def mock_signature(obj):
            sig = original_signature(obj)
            if hasattr(obj, "__name__") and "_build_model" in getattr(obj, "__name__", ""):
                return sig.replace(parameters=[p for p in sig.parameters.values() if p.name != "rules"])
            return sig

        with patch("inspect.signature", side_effect=mock_signature):
            est_loaded = HTSKClassifier.load(path)
            assert est_loaded.model_ is not None


def test_regressor_load_without_rules_support() -> None:
    """Test regressor load path when _build_regressor_model doesn't support rules (line 1032)."""
    import inspect
    import tempfile
    from unittest.mock import patch

    from highfis import HTSKRegressor

    x, y = _make_dataset(20)
    est = HTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est.fit(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.ckpt"
        est.save(path)

        # Mock inspect.signature to simulate _build_regressor_model without 'rules'
        original_signature = inspect.signature

        def mock_signature(obj):
            sig = original_signature(obj)
            if hasattr(obj, "__name__") and "_build_regressor_model" in getattr(obj, "__name__", ""):
                return sig.replace(parameters=[p for p in sig.parameters.values() if p.name != "rules"])
            return sig

        with patch("inspect.signature", side_effect=mock_signature):
            est_loaded = HTSKRegressor.load(path)
            assert est_loaded.model_ is not None


def test_classifier_load_with_consequent_layer_mode_no_method() -> None:
    """Test load when setting consequent_layer.mode directly (lines 842-843)."""
    import tempfile

    x, y = _make_dataset(20)
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est.fit(x, y)

    # Mock the model to have mode in consequent_layer but no set_consequent_mode
    original_has_method = hasattr(est.model_, "set_consequent_mode")
    if original_has_method:
        delattr(est.model_, "set_consequent_mode")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.ckpt"
        est.save(path)
        est_loaded = HTSKClassifier.load(path)
        assert est_loaded.model_ is not None


def test_regressor_load_with_consequent_layer_mode_no_method() -> None:
    """Test regressor load when setting consequent_layer.mode directly (lines 1041-1042)."""
    import tempfile

    from highfis import HTSKRegressor

    x, y = _make_dataset(20)
    est = HTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est.fit(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.ckpt"
        est.save(path)
        est_loaded = HTSKRegressor.load(path)
        assert est_loaded.model_ is not None
        assert est_loaded.n_features_in_ == est.n_features_in_


def test_classifier_save_model_without_rule_layer() -> None:
    """Test save when model doesn't have rule_layer (covers line 789->791 false branch)."""
    import tempfile
    from unittest.mock import patch

    x, y = _make_dataset(20)
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est.fit(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.ckpt"
        # Mock hasattr to return False for rule_layer check
        original_hasattr = hasattr

        def mock_hasattr(obj, name):
            if obj is est.model_ and name == "rule_layer":
                return False
            return original_hasattr(obj, name)

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            est.save(path)
            assert True  # If we reach here, save completed without error


def test_regressor_save_model_without_rule_layer() -> None:
    """Test regressor save when model doesn't have rule_layer (covers line 992->994 false branch)."""
    import tempfile
    from unittest.mock import patch

    from highfis import HTSKRegressor

    x, y = _make_dataset(20)
    est = HTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est.fit(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.ckpt"
        # Mock hasattr to return False for rule_layer check
        original_hasattr = hasattr

        def mock_hasattr(obj, name):
            if obj is est.model_ and name == "rule_layer":
                return False
            return original_hasattr(obj, name)

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            est.save(path)
            assert True  # If we reach here, save completed without error


def test_classifier_load_consequent_mode_elif_branch() -> None:
    """Test elif branch for consequent_layer.mode when set_consequent_mode doesn't exist (line 842-843)."""
    import tempfile
    from unittest.mock import patch

    x, y = _make_dataset(20)
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est.fit(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.ckpt"
        est.save(path)

        # Intercept load to inject a checkpoint with consequent_mode
        from highfis.persistence import load_checkpoint

        original_load_checkpoint = load_checkpoint

        def mock_load_checkpoint(fpath):
            checkpoint = original_load_checkpoint(fpath)
            # Ensure consequent_mode is not None
            checkpoint["model_init"]["consequent_mode"] = "test_mode"
            return checkpoint

        # Mock hasattr to return False for set_consequent_mode
        original_hasattr = hasattr

        def mock_hasattr_no_set_method(obj, name):
            if name == "set_consequent_mode":
                return False
            return original_hasattr(obj, name)

        with (
            patch("highfis.persistence.load_checkpoint", mock_load_checkpoint),
            patch("builtins.hasattr", side_effect=mock_hasattr_no_set_method),
        ):
            est_loaded = HTSKClassifier.load(path)
            assert est_loaded.model_ is not None


def test_regressor_load_consequent_mode_elif_branch() -> None:
    """Test elif branch for consequent_layer.mode when set_consequent_mode doesn't exist (lines 1041-1042)."""
    import tempfile
    from unittest.mock import patch

    from highfis import HTSKRegressor

    x, y = _make_dataset(20)
    est = HTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=1, random_state=42, batch_size=16)
    est.fit(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.ckpt"
        est.save(path)

        # Intercept load to inject a checkpoint with consequent_mode
        from highfis.persistence import load_checkpoint

        original_load_checkpoint = load_checkpoint

        def mock_load_checkpoint(fpath):
            checkpoint = original_load_checkpoint(fpath)
            # Ensure consequent_mode is not None
            checkpoint["model_init"]["consequent_mode"] = "test_mode"
            return checkpoint

        # Mock hasattr to return False for set_consequent_mode
        original_hasattr = hasattr

        def mock_hasattr_selective(obj, name):
            if name == "set_consequent_mode":
                return False
            if name == "mode" and hasattr(obj, "__class__") and "consequent_layer" in str(type(obj)):
                return True
            return original_hasattr(obj, name)

        with (
            patch("highfis.persistence.load_checkpoint", mock_load_checkpoint),
            patch("builtins.hasattr", side_effect=mock_hasattr_selective),
        ):
            est_loaded = HTSKRegressor.load(path)
            assert est_loaded.model_ is not None


def test_classifier_save_load_preserves_classes_dtype_and_scores() -> None:
    """Reloaded classifier keeps ``classes_``/``predict`` dtype so sklearn metrics work.

    Regression for ISSUE_reloaded_classifier_object_dtype.md: ``load`` used to
    restore ``classes_`` as ``object`` dtype, which made ``predict`` return
    ``object`` and ``score``/metrics raise
    "can't handle a mix of multiclass and unknown targets".
    """
    import tempfile

    x, y = _make_dataset(40)
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/clf.pt"
        est.save(path)
        reloaded = HTSKClassifier.load(path)

    assert reloaded.classes_.dtype == est.classes_.dtype
    assert reloaded.predict(x).dtype == est.predict(x).dtype
    # Must not raise (object dtype used to break _check_targets) and match.
    assert reloaded.score(x, y) == est.score(x, y)


def test_rule_base_not_shared_across_cache_key() -> None:
    """CoCo and Cartesian rule bases must yield different rule counts.

    Regression: the MF-initialisation cache key omitted ``rule_base``, so fitting
    ``rule_base="coco"`` and then ``rule_base="cartesian"`` on the same data
    returned the cached CoCo result — Cartesian silently produced S rules instead
    of S**D. The cache key must include ``rule_base``.
    """
    from highfis import TSKClassifier, clear_mf_cache

    x, y = _make_dataset(40)  # D = 3
    clear_mf_cache()

    coco = TSKClassifier(n_mfs=3, mf_init="kmeans", rule_base="coco", epochs=1, random_state=0)
    coco.fit(x, y)
    # Same data/params, only rule_base differs -> must not hit the CoCo cache entry.
    cart = TSKClassifier(n_mfs=3, mf_init="kmeans", rule_base="cartesian", epochs=1, random_state=0)
    cart.fit(x, y)

    assert coco.rule_base_ == "coco"
    assert cart.rule_base_ == "cartesian"
    assert coco.model_.n_rules == 3
    assert cart.model_.n_rules == 3**3  # full Cartesian product over 3 features


def test_mf_cache_lru_eviction_renews_on_access() -> None:
    """A cache hit renews the entry (LRU): the least-recently-used is evicted."""
    from highfis.estimators._base import _MFInitCache

    cache = _MFInitCache(maxsize=2)
    value: tuple[dict[str, Any], list[str], str] = ({}, [], "coco")
    cache.set("a", value)
    cache.set("b", value)
    assert cache.get("a") is not None  # renews "a" -> "b" becomes least-recently-used
    cache.set("c", value)  # evicts "b", not "a"
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
    assert cache.info().currsize == 2


def test_mf_cache_info_and_clear_reset_stats() -> None:
    from highfis.estimators._base import MFCacheInfo, _MFInitCache

    cache = _MFInitCache(maxsize=4)
    value: tuple[dict[str, Any], list[str], str] = ({}, [], "coco")
    assert cache.get("x") is None  # miss
    cache.set("x", value)
    assert cache.get("x") is not None  # hit
    info = cache.info()
    assert (info.hits, info.misses, info.currsize) == (1, 1, 1)
    cache.clear()
    assert cache.info() == MFCacheInfo(hits=0, misses=0, maxsize=4, currsize=0, enabled=True)


def test_mf_cache_set_size_evicts_and_validates() -> None:
    from highfis.estimators._base import _MFInitCache

    cache = _MFInitCache(maxsize=5)
    value: tuple[dict[str, Any], list[str], str] = ({}, [], "coco")
    for k in range(5):
        cache.set(k, value)
    assert cache.info().currsize == 5
    cache.set_maxsize(2)
    assert cache.info().currsize == 2
    with pytest.raises(ValueError, match="maxsize must be >= 1"):
        cache.set_maxsize(0)


def test_set_mf_cache_enabled_bypasses_cache() -> None:
    from highfis import clear_mf_cache, mf_cache_info, set_mf_cache_enabled

    x, _ = _make_dataset(40)
    clear_mf_cache()
    set_mf_cache_enabled(False)
    try:
        est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=7, batch_size=16)
        mfs1, _, _ = est._build_input_mfs(x)
        mfs2, _, _ = est._build_input_mfs(x)
        info = mf_cache_info()
        assert info.enabled is False
        assert info.currsize == 0  # nothing stored while disabled
        # Still produces valid, independent MFs each call.
        assert list(mfs1.keys()) == list(mfs2.keys())
    finally:
        set_mf_cache_enabled(True)  # restore global state for other tests
        clear_mf_cache()


def test_read_cache_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from highfis.estimators._base import _read_cache_env

    monkeypatch.delenv("HIGHFIS_DISABLE_MF_CACHE", raising=False)
    monkeypatch.delenv("HIGHFIS_MF_CACHE_SIZE", raising=False)
    assert _read_cache_env() == {"maxsize": 128, "enabled": True}

    monkeypatch.setenv("HIGHFIS_DISABLE_MF_CACHE", "1")
    assert _read_cache_env()["enabled"] is False
    monkeypatch.setenv("HIGHFIS_DISABLE_MF_CACHE", "no")  # not truthy -> enabled
    assert _read_cache_env()["enabled"] is True

    monkeypatch.setenv("HIGHFIS_MF_CACHE_SIZE", "10")
    assert _read_cache_env()["maxsize"] == 10
    monkeypatch.setenv("HIGHFIS_MF_CACHE_SIZE", "bogus")  # invalid -> default
    assert _read_cache_env()["maxsize"] == 128
    monkeypatch.setenv("HIGHFIS_MF_CACHE_SIZE", "0")  # < 1 -> default
    assert _read_cache_env()["maxsize"] == 128


def test_set_mf_cache_size_changes_capacity_and_evicts() -> None:
    from highfis import clear_mf_cache, mf_cache_info, set_mf_cache_size

    x, _ = _make_dataset(40)
    clear_mf_cache()
    try:
        set_mf_cache_size(4)
        assert mf_cache_info().maxsize == 4
        for rs in range(1, 7):  # six distinct keys into a size-4 cache
            HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, random_state=rs, batch_size=16)._build_input_mfs(x)
        assert mf_cache_info().currsize == 4  # capped, LRU-evicted
    finally:
        set_mf_cache_size(128)  # restore the default for other tests
    assert mf_cache_info().maxsize == 128
