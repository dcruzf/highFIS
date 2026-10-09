"""MHTSK follows its source article (Bian et al., 2025)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from highfis import MHTSKClassifier, MHTSKRegressor
from highfis.estimators._mhtsk import _resolve_mhtsk_scale_parameters

ESTIMATORS = [MHTSKClassifier, MHTSKRegressor]


def _data(n_features: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((60, n_features)).astype(np.float32)
    return x, (x[:, 0] > 0.5).astype(int)


def _fit(estimator_cls: type, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> Any:
    target = y if estimator_cls is MHTSKClassifier else x[:, 0]
    return estimator_cls(random_state=0, **kwargs).fit(x, target)


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_only_the_consequents_are_trained_in_high_dimension(estimator_cls: type) -> None:
    x, y = _data(n_features=1200)
    model = _fit(estimator_cls, x, y, epochs=2, n_heads=5)
    trainable = [name for name, param in model.model_.named_parameters() if param.requires_grad]
    assert trainable == ["consequent_layer.weight", "consequent_layer.bias"]
    sigmas = {round(mf["sigma"], 5) for sets in model.get_mf_params().values() for mf in sets if "sigma" in mf}
    assert sigmas == {1.0}


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_the_antecedents_are_trained_in_low_dimension(estimator_cls: type) -> None:
    """The article only treats more than 1000 features; with few, fixed spreads of one are too wide."""
    x, y = _data()
    model = _fit(estimator_cls, x, y, epochs=2)
    trainable = [name for name, param in model.model_.named_parameters() if param.requires_grad]
    assert "membership_layer._flat_mean" in trainable
    assert "membership_layer._flat_raw_sigma" in trainable


def test_classifier_consequents_start_at_zero() -> None:
    x, y = _data()
    model = _fit(MHTSKClassifier, x, y, epochs=1, learning_rate=0.0)
    assert float(np.abs(model.get_consequent_weights()).max()) == 0.0
    assert float(np.abs(model.get_consequent_bias()).max()) == 0.0


def test_regressor_starts_from_the_target_mean() -> None:
    x, y = _data()
    model = _fit(MHTSKRegressor, x, y, epochs=1, learning_rate=0.0)
    assert float(np.abs(model.get_consequent_weights()).max()) == 0.0
    assert model.predict(x) == pytest.approx(np.full(len(x), x[:, 0].mean()), abs=1e-5)


@pytest.mark.parametrize(("n_features", "expected"), [(2000, (40, 200)), (5000, (100, 200)), (7129, (71, 300))])
def test_scale_parameters_of_the_article_in_high_dimension(n_features: int, expected: tuple[int, int]) -> None:
    resolved = _resolve_mhtsk_scale_parameters(n_features, None, None, None, None, None, 1.0, 743.0)
    assert resolved == expected


def test_number_of_heads_follows_the_coverage_in_low_dimension() -> None:
    head_size, n_heads = _resolve_mhtsk_scale_parameters(40, None, None, None, None, None, 1.0, 743.0)
    assert (head_size, n_heads) == (1, 76)  # a feature coverage of 85%
    assert _resolve_mhtsk_scale_parameters(2000, None, None, None, 0.85, None, 1.0, 743.0)[1] == 95


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
@pytest.mark.parametrize("rule_extraction", [False, True])
def test_predictions_survive_save_and_load(estimator_cls: type, rule_extraction: bool, tmp_path: Path) -> None:
    """Loading a saved MHTSK model used to fail."""
    x, y = _data()
    model = _fit(estimator_cls, x, y, epochs=3, rule_extraction=rule_extraction)
    path = str(tmp_path / "model.pt")
    model.save(path)
    loaded = estimator_cls.load(path)
    np.testing.assert_allclose(loaded.predict(x), model.predict(x), rtol=1e-6, atol=1e-6)
