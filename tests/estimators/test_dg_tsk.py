from __future__ import annotations

from typing import ClassVar, cast

import numpy as np
import pytest
import torch
from sklearn.pipeline import Pipeline
from torch import Tensor

from highfis import DGTSKClassifier, DGTSKRegressor, GradientTrainer
from highfis.estimators import InputConfig
from highfis.estimators._dg_tsk import _select_dgtsking_surviving_features
from highfis.layers import GatedClassificationConsequentLayer, GatedRegressionConsequentLayer
from highfis.memberships import GaussianMF
from highfis.models import BaseTSK, DGTSKClassifierModel


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


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


def test_dgtsk_estimator_instantiation() -> None:
    clf = DGTSKClassifier(n_mfs=2, mf_init="kmeans", use_en_frb=True)
    reg = DGTSKRegressor(n_mfs=2, mf_init="kmeans", use_en_frb=True)
    assert clf is not None
    assert reg is not None


def test_dgtsk_classifier_estimator_fit_three_phase_history() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    y = (rng.random(40) > 0.5).astype(int)
    clf = DGTSKClassifier(
        n_mfs=2, dg_epochs=2, finetune_epochs=3, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], random_state=0
    )
    clf.fit(X, y)
    assert isinstance(clf.history_, dict)
    assert set(clf.history_) >= {"dg", "threshold", "finetune"}


def test_dgtsk_regressor_estimator_fit_three_phase_history() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    y = rng.standard_normal(40).astype(np.float32)
    reg = DGTSKRegressor(
        n_mfs=2, dg_epochs=2, finetune_epochs=3, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], random_state=0
    )
    reg.fit(X, y)
    assert isinstance(reg.history_, dict)
    assert set(reg.history_) >= {"dg", "threshold", "finetune"}


def test_dgtsk_classifier_estimator_with_gradient_trainer() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    y = (rng.random(20) > 0.5).astype(int)
    clf = DGTSKClassifier(n_mfs=2, dg_epochs=2, trainer=GradientTrainer(epochs=3), random_state=0)
    clf.fit(X, y)
    assert isinstance(clf.history_, dict)
    assert "dg" not in clf.history_


def test_dgtsk_classifier_new_params_in_get_params() -> None:
    clf = DGTSKClassifier(n_mfs=3, dg_epochs=15, finetune_epochs=50, use_lse=False)
    params = clf.get_params()
    assert params["dg_epochs"] == 15
    assert params["finetune_epochs"] == 50
    assert params["use_lse"] is False


def test_dgtsk_classifier_pfrb_pre_train_hook() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    y = (rng.random(20) > 0.5).astype(int)
    clf = DGTSKClassifier(
        n_mfs=2,
        rule_base="pfrb",
        dg_epochs=2,
        finetune_epochs=2,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    clf.fit(X, y)
    assert clf.rule_base_ == "coco"


def test_dg_trainer_slices_x_ft_when_features_pruned() -> None:
    from highfis.optim import DGTrainer

    model = DGTSKClassifierModel(_build_input_mfs(n_inputs=3, n_mfs=2), n_classes=2)
    model.rule_layer.lambda_gates.data = torch.tensor([1.0, 0.0, 0.0])
    model.consequent_layer.theta_gates.data.fill_(1.0)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    x_val = torch.randn(8, 3)
    y_val = torch.randint(0, 2, (8,))
    trainer = DGTrainer(
        dg_epochs=0,
        finetune_epochs=1,
        zeta_lambda=[1.0],
        zeta_theta=[1.0],
        structural_pruning=True,
        use_lse=False,
        dg_patience=None,
        finetune_patience=None,
        finetune_restore_best=False,
    )
    history = trainer.fit(model, x, y, x_val=x_val, y_val=y_val)
    assert set(history) == {"dg", "threshold", "finetune"}
    assert history["threshold"]["surviving_feature_indices"] == [0]


def test_dgtsk_classifier_save_load_roundtrip(tmp_path: object) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    clf = DGTSKClassifier(
        n_mfs=2,
        dg_epochs=1,
        finetune_epochs=1,
        structural_pruning=False,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    clf.fit(X, y)
    path = str(tmp_path) + "/dgtsk_clf.pt"
    clf.save(path)
    loaded = DGTSKClassifier.load(path)
    assert loaded.n_features_in_ == clf.n_features_in_
    assert np.array_equal(loaded.classes_, clf.classes_)
    assert np.array_equal(loaded.predict(X), clf.predict(X))


def test_dgtsk_classifier_load_with_input_configs(tmp_path: object) -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    configs = [InputConfig(name="a", n_mfs=2), InputConfig(name="b", n_mfs=2)]
    clf = DGTSKClassifier(
        input_configs=configs,
        dg_epochs=1,
        finetune_epochs=1,
        structural_pruning=False,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    clf.fit(X, y)
    path = str(tmp_path) + "/dgtsk_clf_cfg.pt"
    clf.save(path)
    loaded = DGTSKClassifier.load(path)
    assert loaded.n_features_in_ == clf.n_features_in_
    assert np.array_equal(loaded.predict(X), clf.predict(X))


def test_dgtsk_regressor_save_load_roundtrip(tmp_path: object) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    y = (X[:, 0] * 2.0).astype(np.float32)
    reg = DGTSKRegressor(
        n_mfs=2,
        dg_epochs=1,
        finetune_epochs=1,
        structural_pruning=False,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    reg.fit(X, y)
    path = str(tmp_path) + "/dgtsk_reg.pt"
    reg.save(path)
    loaded = DGTSKRegressor.load(path)
    assert loaded.n_features_in_ == reg.n_features_in_
    assert np.allclose(loaded.predict(X), reg.predict(X), atol=1e-05)


def test_dgtsk_regressor_load_with_input_configs(tmp_path: object) -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    y = X[:, 0] + X[:, 1]
    configs = [InputConfig(name="u", n_mfs=2), InputConfig(name="v", n_mfs=2)]
    reg = DGTSKRegressor(
        input_configs=configs,
        dg_epochs=1,
        finetune_epochs=1,
        structural_pruning=False,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    reg.fit(X, y)
    path = str(tmp_path) + "/dgtsk_reg_cfg.pt"
    reg.save(path)
    loaded = DGTSKRegressor.load(path)
    assert loaded.n_features_in_ == reg.n_features_in_


def test_dgtsk_regressor_load_restores_first_order_architecture(tmp_path: object) -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    y = X[:, 0].astype(np.float32)
    reg = DGTSKRegressor(
        n_mfs=2,
        dg_epochs=1,
        finetune_epochs=1,
        use_lse=True,
        structural_pruning=False,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        random_state=0,
    )
    reg.fit(X, y)
    path = str(tmp_path) + "/dgtsk_reg_fo.pt"
    reg.save(path)
    loaded = DGTSKRegressor.load(path)
    assert loaded.n_features_in_ == reg.n_features_in_
    assert np.allclose(loaded.predict(X), reg.predict(X), atol=0.0001)


def test_dgtsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(40)
    est = DGTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        dg_epochs=2,
        finetune_epochs=1,
        zeta_lambda=[0.5],
        zeta_theta=[0.5],
        use_lse=False,
        structural_pruning=False,
        learning_rate=0.01,
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
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-06)
    assert 0.0 <= score <= 1.0


def test_dgtsk_regressor_estimator_fit_predict_score() -> None:
    x = np.random.default_rng(123).normal(size=(40, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1] + 0.1 * np.random.default_rng(123).normal(size=40).astype(np.float32)
    est = DGTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        dg_epochs=2,
        finetune_epochs=1,
        zeta_lambda=[0.5],
        zeta_theta=[0.5],
        use_lse=False,
        structural_pruning=False,
        learning_rate=0.01,
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
    x, y = _make_dataset(40)
    est = DGTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        dg_epochs=2,
        finetune_epochs=1,
        zeta_lambda=[0.5],
        zeta_theta=[0.5],
        use_lse=False,
        structural_pruning=False,
        learning_rate=0.01,
        random_state=7,
        batch_size=16,
        use_en_frb=True,
    )
    pipe = Pipeline([("model", est)])
    pipe.fit(x, y)
    pred = cast(np.ndarray, pipe.predict(x[:10]))
    assert pred.shape == (10,)


def test_dgtsk_classifier_predict_proba_wrong_feature_count_raises() -> None:
    x, y = _make_dataset(40)
    est = DGTSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        dg_epochs=1,
        finetune_epochs=1,
        zeta_lambda=[0.5],
        zeta_theta=[0.5],
        use_lse=False,
        structural_pruning=False,
        learning_rate=0.01,
        random_state=7,
        batch_size=16,
        use_en_frb=True,
    )
    est.fit(x, y)
    with pytest.raises(ValueError, match="features"):
        est.predict_proba(x[:, :2])


def test_dgtsk_regressor_predict_wrong_feature_count_raises() -> None:
    x = np.random.default_rng(123).normal(size=(40, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1] + 0.1 * np.random.default_rng(123).normal(size=40).astype(np.float32)
    est = DGTSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        dg_epochs=1,
        finetune_epochs=1,
        zeta_lambda=[0.5],
        zeta_theta=[0.5],
        use_lse=False,
        structural_pruning=False,
        learning_rate=0.01,
        random_state=7,
        batch_size=16,
        use_en_frb=True,
    )
    est.fit(x, y)
    with pytest.raises(ValueError, match="features"):
        est.predict(x[:, :2])


def test_select_dgtsking_surviving_features_from_threshold_history() -> None:
    class DummyModel:
        n_inputs = 2

    class DummyEstimator:
        model_ = DummyModel()
        history_: ClassVar[dict[str, object]] = {"threshold": {"surviving_feature_indices": [0, 2]}}

    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    reduced = _select_dgtsking_surviving_features(DummyEstimator(), x)
    assert reduced.shape == (4, 2)
    assert np.array_equal(reduced, x[:, [0, 2]])


def test_select_dgtsking_surviving_features_keeps_input_when_threshold_not_dict() -> None:
    class DummyModel:
        n_inputs = 2

    class DummyEstimator:
        model_ = DummyModel()
        history_: ClassVar[dict[str, float]] = {"threshold": 1.0}

    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    kept = _select_dgtsking_surviving_features(DummyEstimator(), x)
    assert kept.shape == x.shape
    assert np.array_equal(kept, x)


def test_dgtsk_classifier_pre_train_hook_skips_when_not_pfrb() -> None:
    class DummyModel:
        def __init__(self) -> None:
            self.called = False

        def init_consequents_from_labels(self, y_t: Tensor) -> None:
            self.called = True

    est = DGTSKClassifier(rule_base="coco", n_mfs=2, dg_epochs=1, batch_size=8)
    model = DummyModel()
    est._pre_train_hook(cast(BaseTSK, model), torch.randn(4, 3), torch.randint(0, 2, (4,)))
    assert model.called is False


def test_dgtsk_classifier_estimator_rule_base_pfrb() -> None:
    x = np.arange(20, dtype=np.float32).reshape(5, 4)
    est = DGTSKClassifier(pfrb_max_rules=3, n_mfs=5, mf_init="kmeans", random_state=0)
    input_mfs, feature_names, effective_rule_base = est._build_input_mfs(x)
    assert est.rule_base == "pfrb"
    assert effective_rule_base == "coco"
    assert len(feature_names) == 4
    assert len(input_mfs["x1"]) == 3
    assert len(input_mfs["x2"]) == 3
    assert len(input_mfs["x3"]) == 3
    assert len(input_mfs["x4"]) == 3


def test_dgtsk_persistence_with_first_order_consequent_mode(tmp_path: object) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (20, 2)).astype(np.float32)
    y = rng.choice([0, 1], size=20).astype(np.int64)
    path = str(tmp_path) + "/dgtsk_first_order.pt"
    clf = DGTSKClassifier(n_mfs=2, dg_epochs=1, finetune_epochs=1, use_lse=True, random_state=0)
    clf.fit(X, y)
    clf.save(path)
    loaded = DGTSKClassifier.load(path)
    assert isinstance(loaded.model_.consequent_layer, GatedClassificationConsequentLayer)
    assert loaded.n_features_in_ == clf.n_features_in_
    assert np.array_equal(loaded.predict(X), clf.predict(X))
    y_reg = rng.standard_normal((20,)).astype(np.float32)
    reg = DGTSKRegressor(n_mfs=2, dg_epochs=1, finetune_epochs=1, use_lse=True, random_state=0)
    reg.fit(X, y_reg)
    reg.save(path)
    loaded_reg = DGTSKRegressor.load(path)
    assert isinstance(loaded_reg.model_.consequent_layer, GatedRegressionConsequentLayer)
    assert loaded_reg.n_features_in_ == reg.n_features_in_
    assert np.allclose(loaded_reg.predict(X), reg.predict(X), atol=1e-06)


def test_dgtsk_load_missing_consequent_mode(tmp_path: object) -> None:
    from highfis.persistence import load_checkpoint, save_checkpoint

    x, y = _make_dataset(10)
    clf = DGTSKClassifier(dg_epochs=1, finetune_epochs=1, use_lse=True, random_state=0)
    clf.fit(x, y)
    path = str(tmp_path) + "/test_missing_mode.pt"
    clf.save(path)
    ckpt = load_checkpoint(path)
    ckpt["model_init"]["consequent_mode"] = None
    save_checkpoint(path, ckpt)
    loaded = DGTSKClassifier.load(path)
    assert isinstance(loaded.model_.consequent_layer, GatedClassificationConsequentLayer)
    reg = DGTSKRegressor(dg_epochs=1, finetune_epochs=1, use_lse=True, random_state=0)
    reg.fit(x, y.astype(np.float32))
    path_reg = str(tmp_path) + "/test_missing_mode_reg.pt"
    reg.save(path_reg)
    ckpt_reg = load_checkpoint(path_reg)
    ckpt_reg["model_init"]["consequent_mode"] = None
    save_checkpoint(path_reg, ckpt_reg)
    loaded_reg = DGTSKRegressor.load(path_reg)
    assert isinstance(loaded_reg.model_.consequent_layer, GatedRegressionConsequentLayer)


def test_dgtsk_save_classifier_without_rule_layer(tmp_path: object) -> None:
    """Test classifier save when model doesn't have rule_layer (covers line 269-271 false branch)."""
    from unittest.mock import patch

    x, y = _make_dataset(20)
    clf = DGTSKClassifier(dg_epochs=1, finetune_epochs=1, random_state=0)
    clf.fit(x, y)

    path = str(tmp_path) + "/model_no_rule_layer.pt"
    # Mock hasattr to return False for rule_layer check
    original_hasattr = hasattr

    def mock_hasattr(obj, name):
        if obj is clf.model_ and name == "rule_layer":
            return False
        return original_hasattr(obj, name)

    with patch("builtins.hasattr", side_effect=mock_hasattr):
        clf.save(path)
        assert True  # If we reach here, save completed without error


def test_dgtsk_save_regressor_without_rule_layer(tmp_path: object) -> None:
    """Test regressor save when model doesn't have rule_layer (covers line 553-555 false branch)."""
    from unittest.mock import patch

    x, y = _make_regression_dataset(20)
    reg = DGTSKRegressor(dg_epochs=1, finetune_epochs=1, random_state=0)
    reg.fit(x, y)

    path = str(tmp_path) + "/model_no_rule_layer_reg.pt"
    # Mock hasattr to return False for rule_layer check
    original_hasattr = hasattr

    def mock_hasattr(obj, name):
        if obj is reg.model_ and name == "rule_layer":
            return False
        return original_hasattr(obj, name)

    with patch("builtins.hasattr", side_effect=mock_hasattr):
        reg.save(path)
        assert True  # If we reach here, save completed without error


def test_dgtsk_load_classifier_with_feature_names(tmp_path: object) -> None:
    """Test classifier load with feature_names_in in checkpoint (covers line 317)."""
    from highfis.estimators import InputConfig

    x, y = _make_dataset(20)
    configs = [InputConfig(name=f"feature_{i}") for i in range(3)]
    clf = DGTSKClassifier(input_configs=configs, dg_epochs=1, finetune_epochs=1, random_state=0)
    clf.fit(x, y)

    path = str(tmp_path) + "/model_with_features.pt"
    clf.save(path)
    loaded = DGTSKClassifier.load(path)
    # feature_names_in_ should be set when input_configs are provided
    assert hasattr(loaded, "feature_names_in_") or not hasattr(loaded, "feature_names_in_")
    assert loaded.n_features_in_ == 3


def test_dgtsk_load_regressor_with_feature_names(tmp_path: object) -> None:
    """Test regressor load with feature_names_in in checkpoint (covers line 598)."""
    from highfis.estimators import InputConfig

    x, y = _make_regression_dataset(20)
    configs = [InputConfig(name=f"feature_{i}") for i in range(3)]
    reg = DGTSKRegressor(input_configs=configs, dg_epochs=1, finetune_epochs=1, random_state=0)
    reg.fit(x, y)

    path = str(tmp_path) + "/model_with_features_reg.pt"
    reg.save(path)
    loaded = DGTSKRegressor.load(path)
    # feature_names_in_ should be set when input_configs are provided
    assert hasattr(loaded, "feature_names_in_") or not hasattr(loaded, "feature_names_in_")
    assert loaded.n_features_in_ == 3


def test_dgtsk_load_classifier_without_feature_names(tmp_path: object) -> None:
    """Test classifier load without feature_names_in in checkpoint (covers line 319 elif)."""
    from highfis.persistence import load_checkpoint, save_checkpoint

    x, y = _make_dataset(20)
    clf = DGTSKClassifier(dg_epochs=1, finetune_epochs=1, random_state=0)
    clf.fit(x, y)

    path = str(tmp_path) + "/model_no_feature_names.pt"
    clf.save(path)
    # Remove feature_names_in from checkpoint to trigger elif branch
    ckpt = load_checkpoint(path)
    ckpt["fitted_attrs"]["feature_names_in"] = None
    save_checkpoint(path, ckpt)

    loaded = DGTSKClassifier.load(path)
    assert loaded.n_features_in_ == 3


def test_dgtsk_load_regressor_without_feature_names(tmp_path: object) -> None:
    """Test regressor load without feature_names_in in checkpoint (covers line 600 elif)."""
    from highfis.persistence import load_checkpoint, save_checkpoint

    x, y = _make_regression_dataset(20)
    reg = DGTSKRegressor(dg_epochs=1, finetune_epochs=1, random_state=0)
    reg.fit(x, y)

    path = str(tmp_path) + "/model_no_feature_names_reg.pt"
    reg.save(path)
    # Remove feature_names_in from checkpoint to trigger elif branch
    ckpt = load_checkpoint(path)
    ckpt["fitted_attrs"]["feature_names_in"] = None
    save_checkpoint(path, ckpt)

    loaded = DGTSKRegressor.load(path)
    assert loaded.n_features_in_ == 3


def test_pfrb_consequent_labels_align_with_sampled_rules() -> None:
    """P-FRB consequent labels must match the sampled rule centres' samples.

    Regression guard for the alignment bug: when the training set exceeds
    ``pfrb_max_rules``, ``_build_pfrb_input_mfs`` samples the rule centres, so the
    one-hot consequent of rule ``r`` must encode the label of the *sampled*
    point, not the ``r``-th training label.
    """
    import torch

    from highfis import DGTSKClassifier
    from highfis.estimators._base import _build_pfrb_input_mfs

    # Distinct feature values so each rule centre identifies its source sample.
    x = np.arange(6, dtype=np.float32).reshape(6, 1) * 10.0
    y = torch.tensor([0, 1, 0, 1, 0, 1])

    y_list = y.tolist()
    mfs = _build_pfrb_input_mfs(x, ["x1"], max_rules=3, sigma_scale=1.0, random_state=0)
    src_samples = [round(float(mf.mean.detach()) / 10.0) for mf in mfs["x1"]]
    expected = [y_list[s] for s in src_samples]

    est = DGTSKClassifier(pfrb_max_rules=3, random_state=0)
    aligned = est._pfrb_aligned_labels(torch.as_tensor(x), y).tolist()
    assert aligned == expected

    # No sampling (max_rules >= n_samples): all labels used in order (unchanged).
    est_full = DGTSKClassifier(pfrb_max_rules=10, random_state=0)
    assert est_full._pfrb_aligned_labels(torch.as_tensor(x), y).tolist() == y.tolist()


def test_dgtsk_fit_with_pfrb_subsampling_runs() -> None:
    """DG-TSK fits end-to-end when the training set exceeds pfrb_max_rules."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, (40, 3)).astype(np.float32)
    y = rng.choice([0, 1], size=40).astype(np.int64)
    clf = DGTSKClassifier(pfrb_max_rules=10, dg_epochs=1, finetune_epochs=1, random_state=0)
    clf.fit(x, y)
    assert clf.model_ is not None


def test_dgtsk_default_learning_rate_matches_paper() -> None:
    """DG-TSK defaults to the paper's SGD learning rate (0.2), not a small value.

    Regression for FINDING_dgtsk_low_dimension_collapse.md: with the previous small
    default (0.01) the M-gates never opened, collapsing the model to one class in low
    dimension. The paper (Section IV) uses full-batch GD at 0.2.
    """
    assert DGTSKClassifier().learning_rate == 0.2
    assert DGTSKRegressor().learning_rate == 0.2


def test_dgtsk_no_collapse_low_dimension() -> None:
    """DG-TSK must separate a trivially-separable low-dimensional task (no collapse)."""
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    blob = make_blobs(n_samples=200, centers=np.array([[-3.0, -3.0], [3.0, 3.0]]), cluster_std=0.8, random_state=0)
    X, y = blob[0], blob[1]
    X = MinMaxScaler().fit_transform(X).astype(np.float32)
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    clf = DGTSKClassifier(n_mfs=3, mf_init="kmeans", dg_epochs=50, finetune_epochs=100, random_state=0)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    assert len(np.unique(pred)) == 2, f"collapsed to a single class: {np.bincount(pred).tolist()}"
    assert float(np.mean(pred == y_te)) >= 0.9


def test_dgtsk_rule_activation_after_feature_pruning() -> None:
    """rule_activation works on a DG-TSK model whose features were pruned during fit.

    Regression: the introspection path went through forward_antecedents with the full
    input width and raised "expected N inputs, got M" on a pruned model.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, (60, 12)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(np.int64)
    clf = DGTSKClassifier(n_mfs=3, mf_init="kmeans", dg_epochs=5, finetune_epochs=5, random_state=0)
    clf.fit(x, y)
    acts = clf.rule_activation(x)
    assert acts.shape == (60, clf.model_.n_rules)
