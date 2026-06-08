from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from highfis.estimators import DGTSKClassifier, DGTSKRegressor, InputConfig, TSKClassifier, TSKRegressor
from highfis.persistence import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    deserialize_input_mfs,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_payload,
)


def _valid_payload() -> dict[str, object]:
    return {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "estimator_class": "FakeEstimator",
        "estimator_params": {"n_mfs": 1},
        "model_init": {"input_mfs_config": {}, "rule_base": "cartesian"},
        "model_state_dict": {},
        "fitted_attrs": {"n_features_in": 1, "feature_names_in": ["x1"]},
        "history": None,
    }


class TestPersistenceIO:
    def test_roundtrip(self, tmp_path: Path) -> None:
        payload = _valid_payload()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, payload)
        loaded = load_checkpoint(path)
        assert loaded["format"] == CHECKPOINT_FORMAT
        assert loaded["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert loaded["estimator_class"] == "FakeEstimator"
        assert loaded["model_init"]["rule_base"] == "cartesian"

    def test_load_non_dict_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save([1, 2, 3], path)
        with pytest.raises(ValueError, match="invalid checkpoint: expected a dictionary payload"):
            load_checkpoint(path)

    def test_save_checkpoint_creates_parent_dirs(self, tmp_path: Path) -> None:
        payload = _valid_payload()
        path = tmp_path / "nested" / "checkpoint" / "ckpt.pt"
        save_checkpoint(path, payload)
        assert path.exists()
        assert load_checkpoint(path) == payload

    def test_load_checkpoint_uses_weights_only_true(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = _valid_payload()
        path = tmp_path / "ckpt.pt"
        torch.save(payload, path)
        calls: list[dict] = []
        _original_load = torch.load

        def capturing_load(source, map_location, weights_only):
            calls.append({"weights_only": weights_only})
            return _original_load(source, map_location=map_location, weights_only=weights_only)

        monkeypatch.setattr(torch, "load", capturing_load)
        load_checkpoint(path)
        assert calls == [{"weights_only": True}]

    @pytest.mark.parametrize("missing_key", ["estimator_params", "model_init", "model_state_dict", "fitted_attrs"])
    def test_validate_checkpoint_payload_missing_required_keys(self, missing_key: str) -> None:
        payload = _valid_payload()
        payload.pop(missing_key)
        with pytest.raises(ValueError, match=f"invalid checkpoint: missing '{missing_key}'"):
            validate_checkpoint_payload(payload, expected_estimator_class="FakeEstimator")

    def test_validate_checkpoint_payload(self) -> None:
        payload = _valid_payload()
        validate_checkpoint_payload(payload, expected_estimator_class="FakeEstimator")
        payload["format"] = "other"
        with pytest.raises(ValueError, match="invalid checkpoint format"):
            validate_checkpoint_payload(payload, expected_estimator_class="FakeEstimator")
        payload = _valid_payload()
        payload["format_version"] = "999"
        with pytest.raises(ValueError, match="unsupported checkpoint version"):
            validate_checkpoint_payload(payload, expected_estimator_class="FakeEstimator")
        payload = _valid_payload()
        payload["estimator_class"] = "WrongClass"
        with pytest.raises(ValueError, match="checkpoint was created for"):
            validate_checkpoint_payload(payload, expected_estimator_class="FakeEstimator")


def test_deserialize_input_mfs_raises_on_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown membership function type"):
        deserialize_input_mfs({"f1": [{"type": "nonexistent_mf_xyz", "params": {}}]})


class TestEstimatorPersistence:
    def test_tsk_classifier_save_load_roundtrip(self, tmp_path: Path) -> None:
        x = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]], dtype=float)
        y = np.array([0, 1, 0], dtype=int)
        model = TSKClassifier(epochs=1, n_mfs=2, random_state=0, verbose=False)
        model.fit(x, y)
        path = tmp_path / "tsk_classifier.pt"
        model.save(str(path))
        loaded = TSKClassifier.load(str(path))
        assert loaded.n_features_in_ == model.n_features_in_
        assert np.array_equal(loaded.classes_, model.classes_)
        if hasattr(model, "feature_names_in_"):
            assert np.array_equal(
                cast(np.ndarray, loaded.feature_names_in_),
                cast(np.ndarray, model.feature_names_in_),
            )
        else:
            assert not hasattr(loaded, "feature_names_in_")
        assert np.array_equal(loaded.predict(x), model.predict(x))

    def test_tsk_regressor_save_load_roundtrip(self, tmp_path: Path) -> None:
        x = np.array([[0.0], [1.0], [2.0]], dtype=float)
        y = np.array([0.0, 1.0, 2.0], dtype=float)
        model = TSKRegressor(epochs=1, n_mfs=2, random_state=0, verbose=False)
        model.fit(x, y)
        path = tmp_path / "tsk_regressor.pt"
        model.save(str(path))
        loaded = TSKRegressor.load(str(path))
        assert loaded.n_features_in_ == model.n_features_in_
        if hasattr(model, "feature_names_in_"):
            assert np.array_equal(
                cast(np.ndarray, loaded.feature_names_in_),
                cast(np.ndarray, model.feature_names_in_),
            )
        else:
            assert not hasattr(loaded, "feature_names_in_")
        assert np.allclose(loaded.predict(x), model.predict(x), atol=1e-06)

    def test_tsk_classifier_with_input_configs_save_load_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 2)).astype(np.float32)
        y = (x[:, 0] > 0).astype(int)
        configs = [InputConfig(name="a", n_mfs=2), InputConfig(name="b", n_mfs=2)]
        model = TSKClassifier(input_configs=configs, epochs=1, random_state=0, verbose=False)
        model.fit(x, y)
        path = tmp_path / "tsk_clf_inputcfg.pt"
        model.save(str(path))
        loaded = TSKClassifier.load(str(path))
        assert loaded.n_features_in_ == model.n_features_in_
        assert np.array_equal(loaded.predict(x), model.predict(x))

    def test_tsk_regressor_with_input_configs_save_load_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 1)).astype(np.float32)
        y = x[:, 0] * 2.0
        configs = [InputConfig(name="x", n_mfs=2)]
        model = TSKRegressor(input_configs=configs, epochs=1, random_state=0, verbose=False)
        model.fit(x, y)
        path = tmp_path / "tsk_reg_inputcfg.pt"
        model.save(str(path))
        loaded = TSKRegressor.load(str(path))
        assert loaded.n_features_in_ == model.n_features_in_
        assert np.allclose(loaded.predict(x), model.predict(x), atol=1e-06)


def test_persistence_feature_names_in_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Classifier with feature names in checkpoint (covers line 1083 in _base.py)
    x_clf = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    y_clf = np.array([0, 1], dtype=int)
    clf = TSKClassifier(epochs=1, n_mfs=2, random_state=0, verbose=False)
    clf.fit(x_clf, y_clf)
    clf.feature_names_in_ = np.array(["feat1", "feat2"], dtype=object)
    path_clf = tmp_path / "clf_feat.pt"
    clf.save(str(path_clf))
    loaded_clf = TSKClassifier.load(str(path_clf))
    assert hasattr(loaded_clf, "feature_names_in_")
    assert np.array_equal(cast(np.ndarray, loaded_clf.feature_names_in_), ["feat1", "feat2"])

    # Regressor with feature names in checkpoint (covers line 1577 in _base.py)
    x_reg = np.array([[0.0], [1.0]], dtype=float)
    y_reg = np.array([0.0, 1.0], dtype=float)
    reg = TSKRegressor(epochs=1, n_mfs=2, random_state=0, verbose=False)
    reg.fit(x_reg, y_reg)
    reg.feature_names_in_ = np.array(["feat1"], dtype=object)
    path_reg = tmp_path / "reg_feat.pt"
    reg.save(str(path_reg))
    loaded_reg = TSKRegressor.load(str(path_reg))
    assert hasattr(loaded_reg, "feature_names_in_")
    assert np.array_equal(cast(np.ndarray, loaded_reg.feature_names_in_), ["feat1"])

    # Subclass with feature names set during init, checkpoint has None (covers lines 1085 and 1579 in _base.py)
    class CustomTSKClassifier(TSKClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.feature_names_in_ = np.array(["dummy1", "dummy2"], dtype=object)

    class CustomTSKRegressor(TSKRegressor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.feature_names_in_ = np.array(["dummy1"], dtype=object)

    clf_no_feat = TSKClassifier(epochs=1, n_mfs=2, random_state=0, verbose=False)
    clf_no_feat.fit(x_clf, y_clf)
    path_clf_no_feat = tmp_path / "clf_no_feat.pt"
    clf_no_feat.save(str(path_clf_no_feat))

    reg_no_feat = TSKRegressor(epochs=1, n_mfs=2, random_state=0, verbose=False)
    reg_no_feat.fit(x_reg, y_reg)
    path_reg_no_feat = tmp_path / "reg_no_feat.pt"
    reg_no_feat.save(str(path_reg_no_feat))

    monkeypatch.setattr("highfis.estimators._base.validate_checkpoint_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr("highfis.estimators._dg_tsk.validate_checkpoint_payload", lambda *args, **kwargs: None)
    loaded_custom_clf = CustomTSKClassifier.load(str(path_clf_no_feat))
    assert not hasattr(loaded_custom_clf, "feature_names_in_")

    loaded_custom_reg = CustomTSKRegressor.load(str(path_reg_no_feat))
    assert not hasattr(loaded_custom_reg, "feature_names_in_")

    # DGTSKClassifier with feature names in checkpoint (covers line 309 in _dg_tsk.py)
    dg_clf = DGTSKClassifier(dg_epochs=1, finetune_epochs=1, n_mfs=2, random_state=0, verbose=False)
    dg_clf.fit(x_clf, y_clf)
    dg_clf.feature_names_in_ = np.array(["feat1", "feat2"], dtype=object)
    path_dg_clf = tmp_path / "dg_clf_feat.pt"
    dg_clf.save(str(path_dg_clf))
    loaded_dg_clf = DGTSKClassifier.load(str(path_dg_clf))
    assert hasattr(loaded_dg_clf, "feature_names_in_")
    assert np.array_equal(cast(np.ndarray, loaded_dg_clf.feature_names_in_), ["feat1", "feat2"])

    # DGTSKRegressor with feature names in checkpoint (covers line 582 in _dg_tsk.py)
    dg_reg = DGTSKRegressor(dg_epochs=1, finetune_epochs=1, n_mfs=2, random_state=0, verbose=False)
    dg_reg.fit(x_reg, y_reg)
    dg_reg.feature_names_in_ = np.array(["feat1"], dtype=object)
    path_dg_reg = tmp_path / "dg_reg_feat.pt"
    dg_reg.save(str(path_dg_reg))
    loaded_dg_reg = DGTSKRegressor.load(str(path_dg_reg))
    assert hasattr(loaded_dg_reg, "feature_names_in_")
    assert np.array_equal(cast(np.ndarray, loaded_dg_reg.feature_names_in_), ["feat1"])

    # DG subclasses with feature names set during init, checkpoint has None (covers lines 311 and 584 in _dg_tsk.py)
    class CustomDGTSKClassifier(DGTSKClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.feature_names_in_ = np.array(["dummy1", "dummy2"], dtype=object)

    class CustomDGTSKRegressor(DGTSKRegressor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.feature_names_in_ = np.array(["dummy1"], dtype=object)

    dg_clf_no_feat = DGTSKClassifier(dg_epochs=1, finetune_epochs=1, n_mfs=2, random_state=0, verbose=False)
    dg_clf_no_feat.fit(x_clf, y_clf)
    path_dg_clf_no_feat = tmp_path / "dg_clf_no_feat.pt"
    dg_clf_no_feat.save(str(path_dg_clf_no_feat))

    dg_reg_no_feat = DGTSKRegressor(dg_epochs=1, finetune_epochs=1, n_mfs=2, random_state=0, verbose=False)
    dg_reg_no_feat.fit(x_reg, y_reg)
    path_dg_reg_no_feat = tmp_path / "dg_reg_no_feat.pt"
    dg_reg_no_feat.save(str(path_dg_reg_no_feat))

    loaded_custom_dg_clf = CustomDGTSKClassifier.load(str(path_dg_clf_no_feat))
    assert not hasattr(loaded_custom_dg_clf, "feature_names_in_")

    loaded_custom_dg_reg = CustomDGTSKRegressor.load(str(path_dg_reg_no_feat))
    assert not hasattr(loaded_custom_dg_reg, "feature_names_in_")
