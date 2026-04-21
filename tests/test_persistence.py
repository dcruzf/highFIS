from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from highfis.estimators import TSKClassifierEstimator, TSKRegressorEstimator
from highfis.persistence import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_payload,
)


def _valid_payload() -> dict[str, object]:
    return {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "estimator_class": "FakeEstimator",
        "estimator_params": {"n_mfs": 1},
        "model_init": {"input_mfs": {}, "rule_base": "cartesian"},
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
        assert loaded["format_version"] == CHECKPOINT_VERSION
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

    def test_load_checkpoint_falls_back_when_weights_only_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _valid_payload()
        path = tmp_path / "ckpt.pt"
        torch.save(payload, path)

        sentinel = object()

        def fake_load(source, map_location, weights_only=sentinel):
            if weights_only is not sentinel:
                raise TypeError("weights_only not supported")
            return payload

        monkeypatch.setattr(torch, "load", fake_load)
        loaded = load_checkpoint(path)

        assert loaded == payload

    @pytest.mark.parametrize(
        "missing_key",
        ["estimator_params", "model_init", "model_state_dict", "fitted_attrs"],
    )
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


class TestEstimatorPersistence:
    def test_tsk_classifier_save_load_roundtrip(self, tmp_path: Path) -> None:
        x = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]], dtype=float)
        y = np.array([0, 1, 0], dtype=int)

        model = TSKClassifierEstimator(epochs=1, n_mfs=2, random_state=0, verbose=False)
        model.fit(x, y)

        path = tmp_path / "tsk_classifier.pt"
        model.save(str(path))

        loaded = TSKClassifierEstimator.load(str(path))
        assert loaded.n_features_in_ == model.n_features_in_
        assert np.array_equal(loaded.classes_, model.classes_)
        assert np.array_equal(loaded.feature_names_in_, model.feature_names_in_)
        assert np.array_equal(loaded.predict(x), model.predict(x))

    def test_tsk_regressor_save_load_roundtrip(self, tmp_path: Path) -> None:
        x = np.array([[0.0], [1.0], [2.0]], dtype=float)
        y = np.array([0.0, 1.0, 2.0], dtype=float)

        model = TSKRegressorEstimator(epochs=1, n_mfs=2, random_state=0, verbose=False)
        model.fit(x, y)

        path = tmp_path / "tsk_regressor.pt"
        model.save(str(path))

        loaded = TSKRegressorEstimator.load(str(path))
        assert loaded.n_features_in_ == model.n_features_in_
        assert np.array_equal(loaded.feature_names_in_, model.feature_names_in_)
        assert np.allclose(loaded.predict(x), model.predict(x), atol=1e-6)
