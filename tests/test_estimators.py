from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from torch import Tensor, nn

from highfis.base import BaseTSK
from highfis.clustering import FuzzyCMeans, KMeans
from highfis.estimators import (
    ADATSKClassifier,
    ADATSKRegressor,
    ADMTSKClassifier,
    ADMTSKRegressor,
    ADPTSKClassifier,
    ADPTSKRegressor,
    AYATSKClassifier,
    AYATSKRegressor,
    DGALETSKClassifier,
    DGALETSKRegressor,
    DGTSKClassifier,
    DGTSKRegressor,
    DombiTSKClassifier,
    DombiTSKRegressor,
    FSREADATSKClassifier,
    FSREADATSKRegressor,
    HDFISMinClassifier,
    HDFISMinRegressor,
    HDFISProdClassifier,
    HDFISProdRegressor,
    HTSKClassifier,
    HTSKRegressor,
    InputConfig,
    LogTSKClassifier,
    LogTSKRegressor,
    MHTSKClassifier,
    MHTSKRegressor,
    TSKClassifier,
    TSKRegressor,
    feature_coverage_rate,
)
from highfis.estimators._base import (
    _build_fuzzy_c_means_input_mfs,
    _build_gaussian_input_mfs,
    _build_kmeans_input_mfs,
    _build_mhtsk_input_mfs,
    _build_pfrb_input_mfs,
    _extract_mhtsk_rule_indices,
    _mann_whitney_p_value,
    _normalize_importance,
    _rankdata,
    _resolve_mhtsk_scale_parameters,
    _select_rule_indices,
)
from highfis.memberships import DimensionDependentGaussianMF, GaussianMF, GaussianPiMF, MembershipFunction
from highfis.models import HDFISMinClassifierModel, HDFISMinRegressorModel, TSKRegressorModel


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
        x,
        cast(FuzzyCMeans, DummyFuzzyCMeans()),
        sigma_scale=1.0,
        feature_names=feature_names,
        random_state=0,
    )
    assert list(input_mfs.keys()) == feature_names
    assert len(input_mfs["x1"]) == 2
    assert all(isinstance(mf, GaussianMF) for mf in input_mfs["x1"])
    assert input_mfs["x1"][0].sigma > 0.0


def test_htsk_classifier_estimator_fcm_input_initialization() -> None:
    x, y = _make_dataset(40)
    est = HTSKClassifier(
        n_mfs=2,
        mf_init="fcm",
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    assert est.model_.n_rules > 0


def test_htsk_regressor_estimator_fcm_input_initialization() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    est = HTSKRegressor(
        n_mfs=2,
        mf_init="fcm",
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    assert est.model_.n_rules > 0


def test_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = HTSKClassifier(
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


def test_estimator_evaluate_classification_metrics() -> None:
    x, y = _make_dataset(80)
    est = HTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    report = est.evaluate(x, y)

    assert set(report) == {
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    }
    assert 0.0 <= report["accuracy"] <= 1.0


def test_classifier_estimator_pfrb_kmeans_fit_predict_proba() -> None:
    x, y = _make_dataset(40)
    est = HTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        rule_base="pfrb",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    proba = est.predict_proba(x)

    assert proba.shape == (x.shape[0], 2)


def test_classifier_estimator_pfrb_grid_fit_predict_proba() -> None:
    x, y = _make_dataset(40)
    est = HTSKClassifier(
        n_mfs=2,
        mf_init="grid",
        rule_base="pfrb",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    proba = est.predict_proba(x)

    assert proba.shape == (x.shape[0], 2)


def test_hdfis_prod_classifier_estimator_uses_dimension_dependent_mfs() -> None:
    x, y = _make_dataset(40)
    est = HDFISProdClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    assert all(isinstance(mf, DimensionDependentGaussianMF) for mfs in est.model_.input_mfs.values() for mf in mfs)


def test_hdfis_prod_regressor_estimator_uses_dimension_dependent_mfs() -> None:
    x = np.random.RandomState(0).normal(size=(40, 3)).astype(np.float32)
    y = (x[:, 0] * 0.5 + x[:, 1] * -0.2).astype(np.float32)
    est = HDFISProdRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    assert all(isinstance(mf, DimensionDependentGaussianMF) for mfs in est.model_.input_mfs.values() for mf in mfs)


def test_hdfis_min_classifier_estimator_builds_hdfis_min_model() -> None:
    x, y = _make_dataset(40)
    est = HDFISMinClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=2,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    assert isinstance(est.model_, HDFISMinClassifierModel)
    assert all(not p.requires_grad for p in est.model_.membership_layer.parameters())


def test_hdfis_min_regressor_estimator_builds_hdfis_min_model() -> None:
    x = np.random.RandomState(0).normal(size=(40, 3)).astype(np.float32)
    y = (x[:, 0] * 0.5 + x[:, 1] * -0.2).astype(np.float32)
    est = HDFISMinRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=2,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    assert isinstance(est.model_, HDFISMinRegressorModel)
    assert all(not p.requires_grad for p in est.model_.membership_layer.parameters())


def test_ayatsk_classifier_estimator_fit_predict_score() -> None:
    x, y = _make_dataset(80)
    est = AYATSKClassifier(
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


def test_ayatsk_classifier_estimator_fit_predict_score_short() -> None:
    x, y = _make_dataset(80)
    est = AYATSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
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


def test_ayatsk_regressor_estimator_fit_predict() -> None:
    x = np.random.default_rng(123).normal(size=(40, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1]
    est = AYATSKRegressor(
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


def test_dgaletsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = DGALETSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=1.5,
        dg_epochs=5,
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


def test_dgaletsk_regressor_estimator_fit_predict_score() -> None:
    x = np.random.default_rng(123).normal(size=(80, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1] + 0.1 * np.random.default_rng(123).normal(size=80).astype(np.float32)
    est = DGALETSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=1.5,
        dg_epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    pred = est.predict(x)
    score = est.score(x, y)

    assert pred.shape == (x.shape[0],)
    assert isinstance(score, float)


def test_dgtsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = DGTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        dg_epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
        use_en_frb=True,
    )

    est.fit(x, y)
    proba = est.predict_proba(x)
    pred = est.predict(x)
    score = est.score(x, y)

    assert proba.shape == (x.shape[0], 2)
    assert pred.shape == (x.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert 0.0 <= score <= 1.0


def test_dgtsk_regressor_estimator_fit_predict_score() -> None:
    x = np.random.default_rng(123).normal(size=(80, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1] + 0.1 * np.random.default_rng(123).normal(size=80).astype(np.float32)
    est = DGTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        dg_epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
        use_en_frb=True,
    )

    est.fit(x, y)
    pred = est.predict(x)
    score = est.score(x, y)

    assert pred.shape == (x.shape[0],)
    assert isinstance(score, float)


def test_dgtsk_classifier_estimator_pipeline_integration() -> None:
    x, y = _make_dataset(60)
    est = DGTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        dg_epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
        use_en_frb=True,
    )
    pipe = Pipeline([("model", est)])
    pipe.fit(x, y)
    pred = pipe.predict(x[:10])

    assert pred.shape == (10,)


def test_dgtsk_regressor_estimator_save_load_roundtrip() -> None:
    x = np.random.default_rng(123).normal(size=(80, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1] + 0.1 * np.random.default_rng(123).normal(size=80).astype(np.float32)
    est = DGTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        dg_epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
        use_en_frb=True,
    )
    est.fit(x, y)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        est.save(path)
        loaded = DGTSKRegressor.load(path)
        pred = loaded.predict(x)
        assert pred.shape == (x.shape[0],)
    finally:
        Path(path).unlink()


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


def test_dgtsk_classifier_estimator_rule_base_pfrb() -> None:
    x = np.arange(20, dtype=np.float32).reshape(5, 4)
    est = DGTSKClassifier(
        rule_base="pfrb",
        pfrb_max_rules=3,
        n_mfs=5,
        mf_init="kmeans",
        random_state=0,
    )

    input_mfs, feature_names, effective_rule_base = est._build_input_mfs(x)

    assert effective_rule_base == "coco"
    assert len(feature_names) == 4
    assert len(input_mfs["x1"]) == 3
    assert len(input_mfs["x2"]) == 3
    assert len(input_mfs["x3"]) == 3
    assert len(input_mfs["x4"]) == 3


def test_estimator_grid_init_fit_predict() -> None:
    x, y = _make_dataset(80)
    est = HTSKClassifier(
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


def test_fsre_adatsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = FSREADATSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=1.5,
        fs_epochs=5,
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


def test_estimator_predict_proba_requires_fit() -> None:
    x, _ = _make_dataset(10)
    est = HTSKClassifier(n_mfs=2, epochs=1, batch_size=16)
    with pytest.raises(NotFittedError):
        est.predict_proba(x)


def test_estimator_fit_accepts_validation_data_in_fit() -> None:
    x, y = _make_dataset(40)
    x_val, y_val = _make_dataset(10)
    est = HTSKClassifier(n_mfs=2, epochs=1, batch_size=16)
    est = est.fit(x, y, x_val=x_val, y_val=y_val)

    assert hasattr(est, "history_")
    assert "train" in est.history_
    assert "val" in est.history_
    assert est.model_ is not None


def test_estimator_fit_encodes_negative_validation_labels() -> None:
    x, y = _make_dataset(40)
    x_val, y_val = _make_dataset(10)
    y = np.where(y == 0, -1, 1)
    y_val = np.where(y_val == 0, -1, 1)

    est = HTSKClassifier(n_mfs=2, epochs=1, batch_size=16, random_state=0)
    est.fit(x, y, x_val=x_val, y_val=y_val)

    assert np.array_equal(est.classes_, np.array([-1, 1]))


def test_estimator_fit_encodes_string_validation_labels() -> None:
    x, y = _make_dataset(40)
    x_val, y_val = _make_dataset(10)
    y = np.where(y == 0, "neg", "pos")
    y_val = np.where(y_val == 0, "neg", "pos")

    est = HTSKClassifier(n_mfs=2, epochs=1, batch_size=16, random_state=0)
    est.fit(x, y, x_val=x_val, y_val=y_val)

    assert np.array_equal(est.classes_, np.array(["neg", "pos"], dtype=object))


def test_estimator_validates_input_config_length() -> None:
    x, y = _make_dataset(20)
    est = HTSKClassifier(input_configs=[InputConfig(name="x1", n_mfs=2)], batch_size=16)
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
        KMeans(n_clusters=n_clusters, random_state=0),
        sigma_scale=1.0,
        feature_names=feature_names,
        random_state=0,
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

    from highfis.memberships import GaussianMF

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

    from highfis.memberships import GaussianMF

    for name in feature_names:
        for m1, m2 in zip(mfs_1[name], mfs_2[name], strict=False):
            assert isinstance(m1, GaussianMF)
            assert isinstance(m2, GaussianMF)
            # sigma_scale=5 must produce >= sigma_scale=1
            assert float(m2.sigma.detach()) >= float(m1.sigma.detach()) - 1e-6


def test_build_fcm_input_mfs_zero_weight_fallback() -> None:
    x = np.zeros((3, 2), dtype=np.float64)
    feature_names = ["x1", "x2"]

    input_mfs = _build_fuzzy_c_means_input_mfs(
        x,
        FuzzyCMeans(n_clusters=2, random_state=0),
        sigma_scale=1.0,
        feature_names=feature_names,
        random_state=0,
    )

    from highfis.memberships import GaussianMF

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
        x,
        cast(KMeans, DummyKMeans()),
        sigma_scale=1.0,
        feature_names=feature_names,
        random_state=0,
    )

    from highfis.memberships import GaussianMF

    assert list(input_mfs.keys()) == feature_names
    for mfs in input_mfs.values():
        assert len(mfs) == 2
        for mf in mfs:
            assert isinstance(mf, GaussianMF)
            assert float(mf.sigma.detach()) > 0.0


def test_mhtsk_classifier_estimator_fit_predict() -> None:
    x, y = _make_dataset(40)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (x.shape[0],)


def test_mhtsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (x.shape[0],)


def test_mhtsk_classifier_estimator_rule_extraction_reduces_rules() -> None:
    x, y = _make_dataset(40)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        crcr_s=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)

    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0
    assert est.model_.n_rules == len(est._extracted_rule_indices_)


def test_mhtsk_regressor_estimator_rule_extraction_reduces_rules() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)

    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0
    assert est.model_.n_rules == len(est._extracted_rule_indices_)


def test_feature_coverage_rate() -> None:
    assert feature_coverage_rate(10, 2, 3) == pytest.approx(1.0 - (8.0 / 10.0) ** 3)


def test_feature_coverage_rate_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_features must be > 0"):
        feature_coverage_rate(0, 1, 1)
    with pytest.raises(ValueError, match="head_size must be between 1 and n_features"):
        feature_coverage_rate(3, 0, 1)
    with pytest.raises(ValueError, match="head_size must be between 1 and n_features"):
        feature_coverage_rate(3, 4, 1)
    with pytest.raises(ValueError, match="n_heads must be >= 0"):
        feature_coverage_rate(3, 1, -1)


def test_build_mhtsk_input_mfs_validates_inputs() -> None:
    x = np.random.rand(10, 3).astype(np.float32)
    feature_names = ["x1", "x2", "x3"]

    with pytest.raises(ValueError, match="head_size must be between 1 and the number of features"):
        _build_mhtsk_input_mfs(
            x,
            feature_names,
            n_heads=1,
            head_size=4,
            n_clusters=2,
            fcm_m=2.0,
            rule_sigma=1.0,
            instance_sample_fraction=1.0,
            random_state=0,
        )

    with pytest.raises(ValueError, match="n_heads must be > 0"):
        _build_mhtsk_input_mfs(
            x,
            feature_names,
            n_heads=0,
            head_size=1,
            n_clusters=2,
            fcm_m=2.0,
            rule_sigma=1.0,
            instance_sample_fraction=1.0,
            random_state=0,
        )

    with pytest.raises(ValueError, match=r"instance_sample_fraction must be in \(0, 1\]"):
        _build_mhtsk_input_mfs(
            x,
            feature_names,
            n_heads=1,
            head_size=1,
            n_clusters=2,
            fcm_m=2.0,
            rule_sigma=1.0,
            instance_sample_fraction=1.5,
            random_state=0,
        )


def test_build_mhtsk_input_mfs_full_instance_fraction_uses_full_data() -> None:
    x = np.random.rand(10, 3).astype(np.float32)
    feature_names = ["x1", "x2", "x3"]
    _, rules, rule_feature_mask = _build_mhtsk_input_mfs(
        x,
        feature_names,
        n_heads=1,
        head_size=1,
        n_clusters=2,
        fcm_m=2.0,
        rule_sigma=1.0,
        instance_sample_fraction=1.0,
        random_state=0,
    )

    assert len(rules) == 2
    assert rule_feature_mask.shape == (2, 3)


def test_resolve_mhtsk_scale_parameters_uses_paper_defaults() -> None:
    head_size, n_heads = _resolve_mhtsk_scale_parameters(
        n_features=1000,
        head_size=None,
        head_size_ratio=None,
        n_heads=None,
        fcr_target=0.85,
        h_value=None,
        sigma=1.0,
        xi=743.0,
    )

    assert head_size == 20
    assert n_heads == math.ceil(-math.log(1.0 - 0.85) * 1000 / head_size)


def test_resolve_mhtsk_scale_parameters_with_head_size_ratio() -> None:
    head_size, n_heads = _resolve_mhtsk_scale_parameters(
        n_features=1000,
        head_size=None,
        head_size_ratio=0.05,
        n_heads=None,
        fcr_target=None,
        h_value=3.0,
        sigma=1.0,
        xi=743.0,
    )

    assert head_size == 50
    assert n_heads == math.ceil(3.0 * 1000 / head_size)


def test_resolve_mhtsk_scale_parameters_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_features must be > 0"):
        _resolve_mhtsk_scale_parameters(
            n_features=0,
            head_size=None,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=0.85,
            h_value=None,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match="head_size must be between 1 and the number of features"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=11,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=0.85,
            h_value=None,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match="n_heads must be > 0"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=1,
            head_size_ratio=None,
            n_heads=0,
            fcr_target=0.85,
            h_value=None,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match=r"head_size_ratio must be in \(0, 1\]"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=None,
            head_size_ratio=1.5,
            n_heads=None,
            fcr_target=0.85,
            h_value=None,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match=r"fcr_target must be in \(0, 1\)"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=None,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=1.0,
            h_value=None,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match="h_value must be > 0"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=None,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=None,
            h_value=0.0,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match="sigma must be > 0"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=None,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=0.85,
            h_value=None,
            sigma=0.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match="xi must be > 0"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=None,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=0.85,
            h_value=None,
            sigma=1.0,
            xi=0.0,
        )


def test_mhtsk_regressor_estimator_samples_instances() -> None:
    x, y = _make_dataset(40)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        instance_sample_fraction=0.5,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y.astype(np.float32))
    assert est.model_.n_rules > 0


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


def test_extract_mhtsk_rule_indices_supervised() -> None:
    norm_w = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    selected = _extract_mhtsk_rule_indices(norm_w, y, 0.5, 0.5)
    assert len(selected) > 0


def test_extract_mhtsk_rule_indices_empty_rules() -> None:
    norm_w = torch.empty((0, 0), dtype=torch.float32)
    selected = _extract_mhtsk_rule_indices(norm_w, None, 0.5, 0.5)
    assert selected == []


def test_extract_mhtsk_rule_indices_fallback_when_empty() -> None:
    norm_w = torch.tensor([[0.1, 0.2], [0.1, 0.2]], dtype=torch.float32)
    y = torch.tensor([0, 0], dtype=torch.long)
    selected = _extract_mhtsk_rule_indices(norm_w, y, 0.0, 0.0)
    assert len(selected) == 1
    assert selected[0] in {0, 1}


def test_mhtsk_classifier_estimator_rule_extraction_with_validation_data() -> None:
    x, y = _make_dataset(40)
    x_val, y_val = _make_dataset(10)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        crcr_s=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y, x_val=x_val, y_val=y_val)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_regressor_estimator_rule_extraction_with_validation_data() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    x_val, y_val = _make_dataset(10)
    y_val = y_val.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y, x_val=x_val, y_val=y_val)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_classifier_estimator_rule_extraction_without_validation_data() -> None:
    x, y = _make_dataset(40)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        crcr_s=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_regressor_estimator_rule_extraction_without_validation_data() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_classifier_estimator_rule_extraction_without_retraining() -> None:
    x, y = _make_dataset(40)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        crcr_s=0.5,
        retrain_after_extraction=False,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_regressor_estimator_rule_extraction_without_retraining() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        retrain_after_extraction=False,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_classifier_extracted_model_rejects_empty_rule_list() -> None:
    x, y = _make_dataset(20)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    with pytest.raises(ValueError, match="At least one rule must be selected"):
        est._build_extracted_model(est.model_.input_mfs, [])


def test_mhtsk_regressor_extracted_model_rejects_empty_rule_list() -> None:
    x, y = _make_dataset(20)
    y = y.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    with pytest.raises(ValueError, match="At least one rule must be selected"):
        est._build_extracted_model(est.model_.input_mfs, [])


def test_mhtsk_input_builder_rejects_invalid_head_size() -> None:
    x, _ = _make_dataset(10)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=1,
        head_size=0,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    with pytest.raises(ValueError, match="head_size must be between 1 and the number of features"):
        est._build_input_mfs(x)


def test_mhtsk_input_builder_rejects_invalid_n_heads() -> None:
    x, _ = _make_dataset(10)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=0,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    with pytest.raises(ValueError, match="n_heads must be > 0"):
        est._build_input_mfs(x)


def test_mhtsk_input_builder_raises_when_fcm_fails(monkeypatch) -> None:
    x, _ = _make_dataset(10)

    class DummyFuzzyCMeans:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self, x: np.ndarray) -> None:
            self.cluster_centers_ = None

    monkeypatch.setattr("highfis.estimators._base.FuzzyCMeans", DummyFuzzyCMeans)

    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=1,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    with pytest.raises(RuntimeError, match="FuzzyCMeans did not converge to a valid solution"):
        est._build_input_mfs(x)


def test_estimator_invalid_mf_init_raises() -> None:
    x, y = _make_dataset(20)
    est = HTSKClassifier(n_mfs=2, mf_init="random", epochs=1, batch_size=16)
    with pytest.raises(ValueError, match="mf_init"):
        est.fit(x, y)


def test_dombi_tsk_classifier_estimator_fit_predict() -> None:
    x, y = _make_dataset(60)
    est = DombiTSKClassifier(
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


def test_dombi_tsk_regressor_estimator_fit_predict_score() -> None:
    x, y = _make_regression_dataset(80)
    est = DombiTSKRegressor(
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


def test_tsk_classifier_estimator_fit_predict() -> None:
    x, y = _make_dataset(60)
    est = TSKClassifier(
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


def test_tsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = TSKRegressor(
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


def test_tsk_regressor_estimator_save_load_roundtrip() -> None:
    x, y = _make_regression_dataset(80)
    est = TSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        est.save(path)
        loaded = TSKRegressor.load(path)
        pred = loaded.predict(x)
        assert pred.shape == (x.shape[0],)
    finally:
        Path(path).unlink()


def test_tsk_regressor_estimator_invalid_input_configs_length() -> None:
    x, y = _make_regression_dataset(20)
    configs = [InputConfig(name="x1", n_mfs=2)]
    est = TSKRegressor(input_configs=configs, epochs=1, batch_size=8)
    with pytest.raises(ValueError, match="input_configs length"):
        est.fit(x, y)


def test_logtsk_classifier_estimator_fit_predict() -> None:
    x, y = _make_dataset(60)
    est = LogTSKClassifier(
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


def test_logtsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = LogTSKRegressor(
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


def test_fsre_adatsk_classifier_estimator_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_init must be > 0"):
        FSREADATSKClassifier(n_mfs=2, mf_init="kmeans", lambda_init=0.0, fs_epochs=1, batch_size=16)


def test_fsre_adatsk_regressor_estimator_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_init must be > 0"):
        FSREADATSKRegressor(n_mfs=2, mf_init="kmeans", lambda_init=0.0, fs_epochs=1, batch_size=16)


def test_estimator_input_configs_length_validator_regressor() -> None:
    x, y = _make_regression_dataset(20)
    configs = [InputConfig(name="x1", n_mfs=2)]
    est = HTSKRegressor(input_configs=configs, epochs=1, batch_size=8)
    with pytest.raises(ValueError, match="input_configs length"):
        est.fit(x, y)


def test_classifier_estimator_save_load_roundtrip() -> None:
    x, y = _make_dataset(60)
    est = HTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        est.save(path)
        loaded = HTSKClassifier.load(path)
        assert np.array_equal(loaded.classes_, est.classes_)
        assert np.allclose(loaded.predict_proba(x), est.predict_proba(x), atol=1e-6)
    finally:
        Path(path).unlink()


def test_dgaletsk_classifier_estimator_pipeline_integration() -> None:
    x, y = _make_dataset(60)
    est = DGALETSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=1.5,
        dg_epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    pipe = Pipeline([("model", est)])
    pipe.fit(x, y)
    pred = pipe.predict(x[:10])

    assert pred.shape == (10,)


def test_estimators_are_compatible_with_sklearn_cross_val_score_default_scoring() -> None:
    x_clf, y_clf = _make_dataset(45)
    clf = HTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=1,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    clf_scores = cross_val_score(
        clf,
        x_clf,
        y_clf,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
    )
    assert clf_scores.shape == (3,)
    assert np.all(np.isfinite(clf_scores))

    x_reg, y_reg = _make_regression_dataset(45)
    reg = HTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=1,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    reg_scores = cross_val_score(
        reg,
        x_reg,
        y_reg,
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
    )
    assert reg_scores.shape == (3,)
    assert np.all(np.isfinite(reg_scores))


def test_tsk_classifier_estimator_save_load_roundtrip() -> None:
    x, y = _make_dataset(60)
    est = TSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        est.save(path)
        loaded = TSKClassifier.load(path)
        assert np.array_equal(loaded.classes_, est.classes_)
        assert np.allclose(loaded.predict_proba(x), est.predict_proba(x), atol=1e-6)
    finally:
        Path(path).unlink()


def test_tsk_classifier_estimator_invalid_input_configs_length() -> None:
    x, y = _make_dataset(20)
    configs = [InputConfig(name="x1", n_mfs=2)]
    est = TSKClassifier(input_configs=configs, epochs=1, batch_size=8)
    with pytest.raises(ValueError, match="input_configs length"):
        est.fit(x, y)


def test_tsk_classifier_resolve_input_configs_invalid_length() -> None:
    x, _ = _make_dataset(20)
    configs = [InputConfig(name="x1", n_mfs=2)]
    est = TSKClassifier(input_configs=configs, epochs=1, batch_size=8)
    with pytest.raises(ValueError, match="input_configs length"):
        est._resolve_input_configs(x)


def test_tsk_regressor_resolve_input_configs_invalid_length() -> None:
    x, _ = _make_dataset(20)
    configs = [InputConfig(name="x1", n_mfs=2)]
    est = TSKRegressor(input_configs=configs, epochs=1, batch_size=8)
    with pytest.raises(ValueError, match="input_configs length"):
        est._resolve_input_configs(x)


def test_tsk_classifier_estimator_fit_with_input_configs() -> None:
    x, y = _make_dataset(60)
    configs = [InputConfig(name=f"x{i + 1}", n_mfs=2) for i in range(3)]
    est = TSKClassifier(
        input_configs=configs,
        mf_init="grid",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_tsk_regressor_estimator_fit_with_input_configs() -> None:
    x, y = _make_regression_dataset(80)
    configs = [InputConfig(name=f"x{i + 1}", n_mfs=2) for i in range(3)]
    est = TSKRegressor(
        input_configs=configs,
        mf_init="grid",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_logtsk_classifier_estimator_save_load_roundtrip() -> None:
    x, y = _make_dataset(60)
    est = LogTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        est.save(path)
        loaded = LogTSKClassifier.load(path)
        assert np.array_equal(loaded.classes_, est.classes_)
        assert np.allclose(loaded.predict_proba(x), est.predict_proba(x), atol=1e-6)
    finally:
        Path(path).unlink()


def test_estimator_kmeans_default_rule_base_is_coco() -> None:
    """When mf_init='kmeans' and rule_base is not set, model uses 'coco' rule base."""
    x, y = _make_dataset(60)
    est = HTSKClassifier(n_mfs=3, mf_init="kmeans", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)
    # With coco + 3 clusters and 3 features → 3 rules
    assert est.model_.n_rules == 3  # type: ignore[attr-defined]


def test_estimator_early_stopping_with_validation_data() -> None:
    """Estimator stops early when validation_data is provided."""
    x, y = _make_dataset(80)
    x_train, x_val = x[:60], x[60:]
    y_train, y_val = y[:60], y[60:]

    est = HTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=1000,
        learning_rate=5e-2,
        random_state=7,
        patience=3,
    )
    est.fit(x_train, y_train, x_val=x_val, y_val=y_val)

    assert "val" in est.history_
    assert len(est.history_["val"]) > 0
    assert est.history_["stopped_epoch"] < 1000


def test_estimator_no_val_runs_full_epochs() -> None:
    """Without validation_data, training runs for the full epoch count."""
    x, y = _make_dataset(60)
    est = HTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=10,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)

    assert est.history_["stopped_epoch"] == 10


def test_estimator_patience_none_disables_early_stopping() -> None:
    """Setting patience=None should disable early stopping even with validation data."""
    x, y = _make_dataset(80)
    x_train, x_val = x[:60], x[60:]
    y_train, y_val = y[:60], y[60:]

    est = HTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=10,
        learning_rate=5e-2,
        random_state=7,
        patience=None,
    )
    est.fit(x_train, y_train, x_val=x_val, y_val=y_val)

    assert est.history_["stopped_epoch"] == 10
    assert len(est.history_["val"]) == 10


def test_estimator_restore_best_false_does_not_restore_best_model() -> None:
    """With restore_best=False, estimator should keep final weights."""
    x, y = _make_dataset(80)
    x_train, x_val = x[:60], x[60:]
    y_train, y_val = y[:60], y[60:]

    est = HTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
        learning_rate=5e-2,
        random_state=7,
        patience=1,
        restore_best=False,
    )
    est.fit(x_train, y_train, x_val=x_val, y_val=y_val)

    assert est.history_["stopped_epoch"] == 5
    assert len(est.history_["val"]) == 5


def test_estimator_passes_restore_best_to_model_fit() -> None:
    class SpyModel(BaseTSK):
        def __init__(self, input_mfs: Mapping[str, Sequence[MembershipFunction]], rule_base: str) -> None:
            super().__init__(input_mfs, rule_base=rule_base)
            self.fit_kwargs: dict[str, object] | None = None

        def _build_consequent_layer(self) -> nn.Module:
            from torch import nn

            return nn.Linear(self.n_inputs, 1, bias=False)

        def _default_criterion(self) -> nn.Module:
            from torch import nn

            return nn.MSELoss()

        def fit(self, *args: object, **kwargs: object) -> dict[str, list[float]]:
            self.fit_kwargs = kwargs
            return {"train": [], "ur": [], "val": []}

    class SpyEstimator(HTSKClassifier):
        def _build_model(
            self,
            input_mfs: Mapping[str, Sequence[MembershipFunction]],
            n_classes: int,
            rule_base: str,
        ) -> SpyModel:
            return SpyModel(input_mfs, rule_base)

    x, y = _make_dataset(60)
    est = SpyEstimator(
        n_mfs=2,
        mf_init="kmeans",
        epochs=1,
        learning_rate=1e-2,
        random_state=7,
        patience=1,
        restore_best=False,
    )
    est.fit(x, y)

    assert isinstance(est.model_, SpyModel)
    assert est.model_.fit_kwargs is not None
    assert est.model_.fit_kwargs["restore_best"] is False


def test_estimator_sigma_scale_auto() -> None:
    """sigma_scale='auto' uses h=sqrt(D) where D is the number of features."""
    x, y = _make_dataset(60)
    est = HTSKClassifier(
        n_mfs=3,
        mf_init="kmeans",
        sigma_scale="auto",
        epochs=2,
        random_state=0,
        batch_size=16,
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
    x = np.vstack(
        [
            np.zeros((10, 2), dtype=np.float64),
            np.ones((10, 2), dtype=np.float64),
        ]
    )
    mfs = _build_kmeans_input_mfs(
        x,
        KMeans(n_clusters=2, random_state=0),
        sigma_scale=1.0,
        feature_names=["x1", "x2"],
        random_state=0,
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
    est = HTSKClassifier(
        input_configs=configs,
        mf_init="grid",
        epochs=2,
        random_state=0,
        batch_size=16,
    )
    est.fit(x, y)
    assert list(est.feature_names_in_) == ["f0", "f1", "f2"]


def test_estimator_fit_with_input_configs_kmeans_resolve_names() -> None:
    """input_configs set + kmeans init → _resolve_feature_names happy path (line 184)."""
    x, y = _make_dataset(60)
    configs = [InputConfig(name=f"g{i}", n_mfs=3) for i in range(3)]
    est = HTSKClassifier(
        input_configs=configs,
        mf_init="kmeans",
        epochs=2,
        random_state=0,
        batch_size=16,
    )
    est.fit(x, y)
    assert list(est.feature_names_in_) == ["g0", "g1", "g2"]


# ---------------------------------------------------------------------------
# predict_proba feature-count validation
# ---------------------------------------------------------------------------


def test_estimator_predict_proba_wrong_feature_count() -> None:
    """predict_proba with wrong number of features raises ValueError (line 271)."""
    x, y = _make_dataset(40)
    est = HTSKClassifier(n_mfs=2, epochs=2, batch_size=16, random_state=0)
    est.fit(x, y)
    with pytest.raises(ValueError, match="expected"):
        est.predict_proba(x[:, :2])


# ===========================================================================
# HTSKRegressor
# ===========================================================================


def _make_regression_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1] + 0.1 * rng.normal(size=n_samples).astype(np.float32)
    return x, y


def test_regressor_estimator_fit_predict_score() -> None:
    x, y = _make_regression_dataset(80)
    est = HTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    pred = est.predict(x)
    score = est.score(x, y)

    assert pred.shape == (x.shape[0],)
    # R² score from RegressorMixin
    assert isinstance(score, float)


def test_regressor_estimator_evaluate_metrics() -> None:
    x, y = _make_regression_dataset(80)
    est = HTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    report = est.evaluate(x, y)

    assert set(report) == {"mse", "mae", "rmse", "r2"}
    assert report["mse"] >= 0.0
    assert np.isclose(report["rmse"], np.sqrt(report["mse"]))


def test_regressor_estimator_pfrb_kmeans_fit_predict() -> None:
    x, y = _make_regression_dataset(40)
    est = HTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        rule_base="pfrb",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (x.shape[0],)


def test_regressor_estimator_pfrb_grid_fit_predict() -> None:
    x, y = _make_regression_dataset(40)
    est = HTSKRegressor(
        n_mfs=2,
        mf_init="grid",
        rule_base="pfrb",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (x.shape[0],)


def test_regressor_estimator_grid_init_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = HTSKRegressor(
        n_mfs=2,
        mf_init="grid",
        epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (x.shape[0],)


def test_estimator_inspection_methods_for_tsk_classifier() -> None:
    x, y = _make_dataset(40)
    est = TSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
        verbose=False,
    )
    est.fit(x, y)

    info = est.inspect()
    assert info["n_rules"] == est.model_.n_rules
    assert info["n_inputs"] == est.model_.n_inputs
    assert info["feature_names"] == list(est.model_.input_names)
    assert info["rule_base"] == est.rule_base_
    assert info["defuzzifier_type"] == type(est.model_.defuzzifier).__name__
    assert isinstance(info["mf_params"], dict)
    assert isinstance(info["rule_table"], list)
    assert len(info["rule_table"]) == est.model_.n_rules
    assert all("rule_id" in rule for rule in info["rule_table"])
    assert all("type" in mf for mfs in info["mf_params"].values() for mf in mfs)

    activations = est.rule_activation(x[:5])
    assert activations.shape == (5, est.model_.n_rules)
    assert np.all(activations >= 0.0)
    assert np.all(activations <= 1.0)
    assert np.allclose(np.sum(activations, axis=1), 1.0, atol=1e-5)

    importance = est.feature_importance()
    assert importance is not None
    assert importance.shape == (x.shape[1],)
    assert np.all(importance >= 0.0)
    assert np.isclose(np.sum(importance), 1.0)


def test_estimator_inspection_methods_for_tsk_regressor() -> None:
    x, y = _make_regression_dataset(40)
    est = TSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
        verbose=False,
    )
    est.fit(x, y)

    info = est.inspect()
    assert info["n_rules"] == est.model_.n_rules
    assert info["n_inputs"] == est.model_.n_inputs
    assert info["feature_names"] == list(est.model_.input_names)
    assert info["rule_base"] == est.rule_base_

    activations = est.rule_activation(x[:5])
    assert activations.shape == (5, est.model_.n_rules)
    assert np.allclose(np.sum(activations, axis=1), 1.0, atol=1e-5)

    importance = est.feature_importance()
    assert importance is not None
    assert importance.shape == (x.shape[1],)
    assert np.all(importance >= 0.0)
    assert np.isclose(np.sum(importance), 1.0)


def test_classifier_fit_requires_validation_inputs_together() -> None:
    x, y = _make_dataset(20)
    est = TSKClassifier(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    with pytest.raises(ValueError, match="x_val and y_val must be provided together"):
        est.fit(x, y, x_val=x, y_val=None)


def test_regressor_fit_requires_validation_inputs_together() -> None:
    x, y = _make_regression_dataset(20)
    est = TSKRegressor(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    with pytest.raises(ValueError, match="x_val and y_val must be provided together"):
        est.fit(x, y, x_val=x, y_val=None)


def test_classifier_rule_activation_validates_feature_count() -> None:
    x, y = _make_dataset(20)
    est = TSKClassifier(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    with pytest.raises(ValueError, match=r"expected .* features, got"):
        est.rule_activation(x[:, :2])


def test_regressor_rule_activation_validates_feature_count() -> None:
    x, y = _make_regression_dataset(20)
    est = TSKRegressor(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    with pytest.raises(ValueError, match=r"expected .* features, got"):
        est.rule_activation(x[:, :2])


def test_normalize_importance_returns_uniform_distribution_for_zero_total() -> None:
    values = torch.zeros(3, dtype=torch.float32)
    normalized = _normalize_importance(values)
    assert normalized.shape == (3,)
    assert np.allclose(normalized, np.array([1.0 / 3.0] * 3))


def test_classifier_feature_importance_returns_none_when_consequent_weights_missing() -> None:
    x, y = _make_dataset(20)
    est = TSKClassifier(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    est.model_.get_consequent_weights = cast(Any, lambda: None)
    assert est.feature_importance() is None


def test_regressor_feature_importance_returns_none_when_consequent_weights_missing() -> None:
    x, y = _make_regression_dataset(20)
    est = TSKRegressor(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    est.model_.get_consequent_weights = cast(Any, lambda: None)
    assert est.feature_importance() is None


def test_classifier_feature_importance_handles_3d_weights_and_mask() -> None:
    x, y = _make_dataset(20)
    est = TSKClassifier(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    est.model_.get_consequent_weights = cast(Any, lambda: torch.ones(2, 3, x.shape[1], dtype=torch.float32))
    est.model_.consequent_layer.rule_feature_mask = torch.ones(3, dtype=torch.float32)  # type: ignore[attr-defined]
    importance = est.feature_importance()
    assert importance is not None
    assert importance.shape == (x.shape[1],)
    assert np.isclose(np.sum(importance), 1.0)


def test_classifier_feature_importance_handles_2d_weights() -> None:
    x, y = _make_dataset(20)
    est = TSKClassifier(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    est.model_.get_consequent_weights = cast(Any, lambda: torch.ones(3, x.shape[1], dtype=torch.float32))
    importance = est.feature_importance()
    assert importance is not None
    assert importance.shape == (x.shape[1],)
    assert np.isclose(np.sum(importance), 1.0)


def test_classifier_feature_importance_raises_for_unsupported_weight_shape() -> None:
    x, y = _make_dataset(20)
    est = TSKClassifier(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    est.model_.get_consequent_weights = cast(Any, lambda: torch.ones(1, 2, 3, 4, dtype=torch.float32))
    with pytest.raises(ValueError, match="unsupported consequent weight shape"):
        est.feature_importance()


def test_regressor_feature_importance_raises_for_unsupported_weight_shape() -> None:
    x, y = _make_regression_dataset(20)
    est = TSKRegressor(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    est.model_.get_consequent_weights = cast(Any, lambda: torch.ones(1, 2, 3, 4, dtype=torch.float32))
    with pytest.raises(ValueError, match="unsupported consequent weight shape"):
        est.feature_importance()


def test_base_get_consequent_weights_returns_none_when_no_weight_attribute() -> None:
    model = TSKRegressorModel({"x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.5)]})
    if hasattr(model.consequent_layer, "weight"):
        delattr(model.consequent_layer, "weight")
    assert model.get_consequent_weights() is None


def test_base_get_consequent_weights_returns_tensor_when_weight_exists() -> None:
    model = TSKRegressorModel({"x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.5)]})
    weight = model.get_consequent_weights()
    assert weight is None or isinstance(weight, torch.Tensor)


def test_regressor_feature_importance_handles_3d_weights_and_mask() -> None:
    x, y = _make_regression_dataset(20)
    est = TSKRegressor(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    est.model_.get_consequent_weights = cast(Any, lambda: torch.ones(2, 3, x.shape[1], dtype=torch.float32))
    est.model_.consequent_layer.rule_feature_mask = torch.ones(3, dtype=torch.float32)  # type: ignore[attr-defined]
    importance = est.feature_importance()
    assert importance is not None
    assert importance.shape == (x.shape[1],)
    assert np.isclose(np.sum(importance), 1.0)


def test_estimator_inspection_methods_for_mhtsk_classifier() -> None:
    x, y = _make_dataset(40)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=2,
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
        verbose=False,
    )
    est.fit(x, y)

    info = est.inspect()
    assert info["n_rules"] == est.model_.n_rules
    assert len(info["rule_table"]) == est.model_.n_rules

    activations = est.rule_activation(x[:5])
    assert activations.shape == (5, est.model_.n_rules)
    assert np.allclose(np.sum(activations, axis=1), 1.0, atol=1e-5)

    importance = est.feature_importance()
    assert importance is not None
    assert importance.shape == (x.shape[1],)
    assert np.all(importance >= 0.0)
    assert np.isclose(np.sum(importance), 1.0)


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


def test_fsre_adatsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = FSREADATSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=2.0,
        fs_epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (x.shape[0],)


def test_regressor_estimator_predict_requires_fit() -> None:
    x, _ = _make_regression_dataset(10)
    est = HTSKRegressor(n_mfs=2, epochs=1, batch_size=16)
    with pytest.raises(NotFittedError):
        est.predict(x)


def test_regressor_estimator_validates_input_config_length() -> None:
    x, y = _make_regression_dataset(20)
    est = HTSKRegressor(input_configs=[InputConfig(name="x1", n_mfs=2)], batch_size=16)
    with pytest.raises(ValueError, match="input_configs length"):
        est.fit(x, y)


def test_regressor_estimator_invalid_mf_init_raises() -> None:
    x, y = _make_regression_dataset(20)
    est = HTSKRegressor(n_mfs=2, mf_init="random", epochs=1, batch_size=16)
    with pytest.raises(ValueError, match="mf_init"):
        est.fit(x, y)


def test_regressor_estimator_kmeans_default_rule_base_is_coco() -> None:
    x, y = _make_regression_dataset(60)
    est = HTSKRegressor(n_mfs=3, mf_init="kmeans", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)
    assert est.model_.n_rules == 3  # type: ignore[attr-defined]


def test_regressor_estimator_early_stopping_with_validation_data() -> None:
    x, y = _make_regression_dataset(80)
    x_train, x_val = x[:60], x[60:]
    y_train, y_val = y[:60], y[60:]

    est = HTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=1000,
        learning_rate=5e-2,
        random_state=7,
        patience=3,
    )
    est.fit(x_train, y_train, x_val=x_val, y_val=y_val)

    assert "val" in est.history_
    assert len(est.history_["val"]) > 0
    assert est.history_["stopped_epoch"] < 1000


def test_regressor_estimator_no_val_runs_full_epochs() -> None:
    x, y = _make_regression_dataset(60)
    est = HTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=10,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)

    assert est.history_["stopped_epoch"] == 10
    assert len(est.history_["val"]) == 0


def test_regressor_estimator_sigma_scale_auto() -> None:
    x, y = _make_regression_dataset(60)
    est = HTSKRegressor(
        n_mfs=3,
        mf_init="kmeans",
        sigma_scale="auto",
        epochs=2,
        random_state=0,
        batch_size=16,
    )
    est.fit(x, y)

    assert est.model_ is not None
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_regressor_estimator_predict_wrong_feature_count() -> None:
    x, y = _make_regression_dataset(40)
    est = HTSKRegressor(n_mfs=2, epochs=2, batch_size=16, random_state=0)
    est.fit(x, y)
    with pytest.raises(ValueError, match="expected"):
        est.predict(x[:, :2])


def test_regressor_estimator_fit_with_input_configs_grid() -> None:
    x, y = _make_regression_dataset(60)
    configs = [InputConfig(name=f"f{i}", n_mfs=2) for i in range(3)]
    est = HTSKRegressor(
        input_configs=configs,
        mf_init="grid",
        epochs=2,
        random_state=0,
        batch_size=16,
    )
    est.fit(x, y)
    assert list(est.feature_names_in_) == ["f0", "f1", "f2"]


def test_regressor_estimator_fit_with_input_configs_kmeans() -> None:
    x, y = _make_regression_dataset(60)
    configs = [InputConfig(name=f"g{i}", n_mfs=3) for i in range(3)]
    est = HTSKRegressor(
        input_configs=configs,
        mf_init="kmeans",
        epochs=2,
        random_state=0,
        batch_size=16,
    )
    est.fit(x, y)
    assert list(est.feature_names_in_) == ["g0", "g1", "g2"]
