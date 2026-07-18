from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from highfis import HTSKClassifier, HTSKRegressor, TSKClassifier, TSKRegressor
from highfis.clustering import KMeans
from highfis.estimators import InputConfig
from highfis.estimators._base import _build_gaussian_input_mfs, _build_kmeans_input_mfs


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return (x, y)


def _make_regression_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1] + 0.1 * rng.normal(size=n_samples).astype(np.float32)
    return (x, y)


def test_htsk_classifier_estimator_fcm_input_initialization() -> None:
    x, y = _make_dataset(40)
    est = HTSKClassifier(n_mfs=2, mf_init="fcm", epochs=1, batch_size=16, random_state=0)
    est.fit(x, y)
    assert est.model_.n_rules > 0


def test_htsk_regressor_estimator_fcm_input_initialization() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    est = HTSKRegressor(n_mfs=2, mf_init="fcm", epochs=1, batch_size=16, random_state=0)
    est.fit(x, y)
    assert est.model_.n_rules > 0


def test_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    proba = est.predict_proba(x)
    pred = est.predict(x)
    score = est.score(x, y)
    assert proba.shape == (x.shape[0], 2)
    assert pred.shape == (x.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-06)
    assert 0.0 <= score <= 1.0


def test_estimator_evaluate_classification_metrics() -> None:
    x, y = _make_dataset(80)
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    report = est.evaluate(x, y)
    assert set(report) == {
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "confusion_matrix",
        "classes",
    }
    assert 0.0 <= report["accuracy"] <= 1.0


def test_classifier_estimator_pfrb_kmeans_fit_predict_proba() -> None:
    x, y = _make_dataset(40)
    est = HTSKClassifier(
        n_mfs=2, mf_init="kmeans", rule_base="pfrb", epochs=3, learning_rate=0.01, random_state=7, batch_size=16
    )
    est.fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (x.shape[0], 2)


def test_classifier_estimator_pfrb_grid_fit_predict_proba() -> None:
    x, y = _make_dataset(40)
    est = HTSKClassifier(
        n_mfs=2, mf_init="grid", rule_base="pfrb", epochs=3, learning_rate=0.01, random_state=7, batch_size=16
    )
    est.fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (x.shape[0], 2)


def test_estimator_grid_init_fit_predict() -> None:
    x, y = _make_dataset(80)
    est = HTSKClassifier(n_mfs=2, mf_init="grid", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    proba = est.predict_proba(x)
    pred = est.predict(x)
    assert proba.shape == (x.shape[0], 2)
    assert pred.shape == (x.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-06)


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
    assert "train_total_loss" in est.history_
    assert "val_loss" in est.history_
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


def test_estimator_invalid_mf_init_raises() -> None:
    x, y = _make_dataset(20)
    est = HTSKClassifier(n_mfs=2, mf_init="random", epochs=1, batch_size=16)
    with pytest.raises(ValueError, match="mf_init"):
        est.fit(x, y)


def test_classifier_estimator_save_load_roundtrip() -> None:
    x, y = _make_dataset(60)
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        est.save(path)
        loaded = HTSKClassifier.load(path)
        assert np.array_equal(loaded.classes_, est.classes_)
        assert np.allclose(loaded.predict_proba(x), est.predict_proba(x), atol=1e-06)
    finally:
        Path(path).unlink()


def test_estimators_are_compatible_with_sklearn_cross_val_score_default_scoring() -> None:
    x_clf, y_clf = _make_dataset(45)
    clf = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1, learning_rate=0.01, random_state=7, batch_size=16)
    clf_scores = cross_val_score(clf, x_clf, y_clf, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0))
    assert clf_scores.shape == (3,)
    assert np.all(np.isfinite(clf_scores))
    x_reg, y_reg = _make_regression_dataset(45)
    reg = HTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=1, learning_rate=0.01, random_state=7, batch_size=16)
    reg_scores = cross_val_score(reg, x_reg, y_reg, cv=KFold(n_splits=3, shuffle=True, random_state=0))
    assert reg_scores.shape == (3,)
    assert np.all(np.isfinite(reg_scores))


def test_estimator_kmeans_default_rule_base_is_coco() -> None:
    x, y = _make_dataset(60)
    est = HTSKClassifier(n_mfs=3, mf_init="kmeans", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)
    assert est.model_.n_rules == 3


def test_estimator_early_stopping_with_validation_data() -> None:
    x, y = _make_dataset(80)
    x_train, x_val = (x[:60], x[60:])
    y_train, y_val = (y[:60], y[60:])
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=1000, learning_rate=0.05, random_state=7, patience=3)
    est.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    assert "val_loss" in est.history_
    assert len(est.history_["val_loss"]) > 0
    assert est.history_["stopped_epoch"] < 1000


def test_estimator_no_val_runs_full_epochs() -> None:
    x, y = _make_dataset(60)
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=10, random_state=7, batch_size=16)
    est.fit(x, y)
    assert est.history_["stopped_epoch"] == 10


def test_estimator_patience_none_disables_early_stopping() -> None:
    x, y = _make_dataset(80)
    x_train, x_val = (x[:60], x[60:])
    y_train, y_val = (y[:60], y[60:])
    est = HTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=10, learning_rate=0.05, random_state=7, patience=None)
    est.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    assert est.history_["stopped_epoch"] == 10
    assert len(est.history_["val_loss"]) == 10


def test_estimator_restore_best_false_does_not_restore_best_model() -> None:
    x, y = _make_dataset(80)
    x_train, x_val = (x[:60], x[60:])
    y_train, y_val = (y[:60], y[60:])
    est = HTSKClassifier(
        n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.05, random_state=7, patience=1, restore_best=False
    )
    est.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    assert est.history_["stopped_epoch"] == 5
    assert len(est.history_["val_loss"]) == 5


def test_estimator_passes_restore_best_to_model_fit() -> None:
    """Verify that the estimator passes restore_best to GradientTrainer."""
    import highfis.estimators._base as _base_module

    captured_kwargs: dict[str, object] = {}

    original_trainer_cls = _base_module.GradientTrainer

    class SpyTrainer(original_trainer_cls):
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            super().__init__(**kwargs)

    _base_module.GradientTrainer = SpyTrainer  # type: ignore
    try:
        x, y = _make_dataset(60)
        est = HTSKClassifier(
            n_mfs=2, mf_init="kmeans", epochs=1, learning_rate=0.01, random_state=7, patience=1, restore_best=False
        )
        est.fit(x, y)
    finally:
        _base_module.GradientTrainer = original_trainer_cls  # type: ignore[invalid-assignment]

    assert captured_kwargs.get("restore_best") is False


def test_estimator_sigma_scale_auto() -> None:
    x, y = _make_dataset(60)
    est = HTSKClassifier(n_mfs=3, mf_init="kmeans", sigma_scale="auto", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)
    assert est.model_ is not None
    proba = est.predict_proba(x)
    assert proba.shape == (x.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-06)


def test_build_gaussian_input_mfs_constant_column() -> None:
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(20), rng.normal(size=20)]).astype(np.float64)
    configs = [InputConfig(name="x1", n_mfs=2), InputConfig(name="x2", n_mfs=2)]
    mfs = _build_gaussian_input_mfs(x, configs)
    assert len(mfs["x1"]) == 2


def test_build_gaussian_input_mfs_single_mf() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 1)).astype(np.float64)
    configs = [InputConfig(name="x1", n_mfs=1)]
    mfs = _build_gaussian_input_mfs(x, configs)
    assert len(mfs["x1"]) == 1
    assert float(mfs["x1"][0].sigma.detach()) > 0


def test_build_kmeans_zero_sigma_fallback() -> None:
    x = np.vstack([np.zeros((10, 2), dtype=np.float64), np.ones((10, 2), dtype=np.float64)])
    mfs = _build_kmeans_input_mfs(
        x, KMeans(n_clusters=2, random_state=0), sigma_scale=1.0, feature_names=["x1", "x2"], random_state=0
    )
    for name in ["x1", "x2"]:
        for mf in mfs[name]:
            assert float(mf.sigma.detach()) > 0


def test_estimator_fit_with_input_configs_grid_resolve_config() -> None:
    x, y = _make_dataset(60)
    configs = [InputConfig(name=f"f{i}", n_mfs=2) for i in range(3)]
    est = HTSKClassifier(input_configs=configs, mf_init="grid", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)
    # input_configs names flow into the model's MFs, not feature_names_in_
    # (feature_names_in_ is only set by sklearn when fit receives a DataFrame)
    assert not hasattr(est, "feature_names_in_")
    assert list(est.model_.input_mfs.keys()) == ["f0", "f1", "f2"]


def test_estimator_fit_with_input_configs_kmeans_resolve_names() -> None:
    x, y = _make_dataset(60)
    configs = [InputConfig(name=f"g{i}", n_mfs=3) for i in range(3)]
    est = HTSKClassifier(input_configs=configs, mf_init="kmeans", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)
    # input_configs names flow into the model's MFs, not feature_names_in_
    assert not hasattr(est, "feature_names_in_")
    assert list(est.model_.input_mfs.keys()) == ["g0", "g1", "g2"]


def test_estimator_predict_proba_wrong_feature_count() -> None:
    x, y = _make_dataset(40)
    est = HTSKClassifier(n_mfs=2, epochs=2, batch_size=16, random_state=0)
    est.fit(x, y)
    with pytest.raises(ValueError, match=r"(?:expected|is expecting)"):
        est.predict_proba(x[:, :2])


def test_regressor_estimator_fit_predict_score() -> None:
    x, y = _make_regression_dataset(80)
    est = HTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    pred = est.predict(x)
    score = est.score(x, y)
    assert pred.shape == (x.shape[0],)
    assert isinstance(score, float)


def test_regressor_estimator_evaluate_metrics() -> None:
    x, y = _make_regression_dataset(80)
    est = HTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    report = est.evaluate(x, y)
    assert set(report) == {
        "mse",
        "mae",
        "rmse",
        "median_absolute_error",
        "mean_bias_error",
        "max_error",
        "std_error",
        "explained_variance",
        "mape",
        "smape",
        "msle",
        "pearson",
        "r2",
    }
    assert report["mse"] >= 0.0
    assert np.isclose(report["rmse"], np.sqrt(report["mse"]))


def test_regressor_estimator_pfrb_kmeans_fit_predict() -> None:
    x, y = _make_regression_dataset(40)
    est = HTSKRegressor(
        n_mfs=2, mf_init="kmeans", rule_base="pfrb", epochs=3, learning_rate=0.01, random_state=7, batch_size=16
    )
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_regressor_estimator_pfrb_grid_fit_predict() -> None:
    x, y = _make_regression_dataset(40)
    est = HTSKRegressor(
        n_mfs=2, mf_init="grid", rule_base="pfrb", epochs=3, learning_rate=0.01, random_state=7, batch_size=16
    )
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_regressor_estimator_grid_init_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = HTSKRegressor(n_mfs=2, mf_init="grid", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_estimator_inspection_methods_for_tsk_classifier() -> None:
    x, y = _make_dataset(40)
    est = TSKClassifier(
        n_mfs=2, mf_init="kmeans", epochs=3, learning_rate=0.01, random_state=7, batch_size=16, verbose=False
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
    assert np.allclose(np.sum(activations, axis=1), 1.0, atol=1e-05)
    importance = est.feature_importance()
    assert importance is not None
    assert importance.shape == (x.shape[1],)
    assert np.all(importance >= 0.0)
    assert np.isclose(np.sum(importance), 1.0)


def test_estimator_inspection_methods_for_tsk_regressor() -> None:
    x, y = _make_regression_dataset(40)
    est = TSKRegressor(
        n_mfs=2, mf_init="kmeans", epochs=3, learning_rate=0.01, random_state=7, batch_size=16, verbose=False
    )
    est.fit(x, y)
    info = est.inspect()
    assert info["n_rules"] == est.model_.n_rules
    assert info["n_inputs"] == est.model_.n_inputs
    assert info["feature_names"] == list(est.model_.input_names)
    assert info["rule_base"] == est.rule_base_
    activations = est.rule_activation(x[:5])
    assert activations.shape == (5, est.model_.n_rules)
    assert np.allclose(np.sum(activations, axis=1), 1.0, atol=1e-05)
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
    with pytest.raises(ValueError, match=r"(?:expected .* features, got|is expecting)"):
        est.rule_activation(x[:, :2])


def test_regressor_rule_activation_validates_feature_count() -> None:
    x, y = _make_regression_dataset(20)
    est = TSKRegressor(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    with pytest.raises(ValueError, match=r"(?:expected .* features, got|is expecting)"):
        est.rule_activation(x[:, :2])


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
    est.model_.consequent_layer.rule_feature_mask = torch.ones(3, dtype=torch.float32)
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


def test_regressor_feature_importance_handles_3d_weights_and_mask() -> None:
    x, y = _make_regression_dataset(20)
    est = TSKRegressor(n_mfs=2, epochs=1, batch_size=5, random_state=0)
    est.fit(x, y)
    est.model_.get_consequent_weights = cast(Any, lambda: torch.ones(2, 3, x.shape[1], dtype=torch.float32))
    est.model_.consequent_layer.rule_feature_mask = torch.ones(3, dtype=torch.float32)
    importance = est.feature_importance()
    assert importance is not None
    assert importance.shape == (x.shape[1],)
    assert np.isclose(np.sum(importance), 1.0)


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
    assert est.model_.n_rules == 3


def test_regressor_estimator_early_stopping_with_validation_data() -> None:
    x, y = _make_regression_dataset(80)
    x_train, x_val = (x[:60], x[60:])
    y_train, y_val = (y[:60], y[60:])
    est = HTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=1000, learning_rate=0.05, random_state=7, patience=3)
    est.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    assert "val_loss" in est.history_
    assert len(est.history_["val_loss"]) > 0
    assert est.history_["stopped_epoch"] < 1000


def test_regressor_estimator_no_val_runs_full_epochs() -> None:
    x, y = _make_regression_dataset(60)
    est = HTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=10, random_state=7, batch_size=16)
    est.fit(x, y)
    assert est.history_["stopped_epoch"] == 10
    assert "val" not in est.history_


def test_regressor_estimator_sigma_scale_auto() -> None:
    x, y = _make_regression_dataset(60)
    est = HTSKRegressor(n_mfs=3, mf_init="kmeans", sigma_scale="auto", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)
    assert est.model_ is not None
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_regressor_estimator_predict_wrong_feature_count() -> None:
    x, y = _make_regression_dataset(40)
    est = HTSKRegressor(n_mfs=2, epochs=2, batch_size=16, random_state=0)
    est.fit(x, y)
    with pytest.raises(ValueError, match=r"(?:expected|is expecting)"):
        est.predict(x[:, :2])


def test_regressor_estimator_fit_with_input_configs_grid() -> None:
    x, y = _make_regression_dataset(60)
    configs = [InputConfig(name=f"f{i}", n_mfs=2) for i in range(3)]
    est = HTSKRegressor(input_configs=configs, mf_init="grid", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)
    # input_configs names flow into the model's MFs, not feature_names_in_
    assert not hasattr(est, "feature_names_in_")
    assert list(est.model_.input_mfs.keys()) == ["f0", "f1", "f2"]


def test_regressor_estimator_fit_with_input_configs_kmeans() -> None:
    x, y = _make_regression_dataset(60)
    configs = [InputConfig(name=f"g{i}", n_mfs=3) for i in range(3)]
    est = HTSKRegressor(input_configs=configs, mf_init="kmeans", epochs=2, random_state=0, batch_size=16)
    est.fit(x, y)
    # input_configs names flow into the model's MFs, not feature_names_in_
    assert not hasattr(est, "feature_names_in_")
    assert list(est.model_.input_mfs.keys()) == ["g0", "g1", "g2"]
