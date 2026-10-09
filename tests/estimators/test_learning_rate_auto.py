"""A learning rate that follows the number of features, for the families that need one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone

import highfis

RULES = {
    "ADATSK": {4: 0.1, 200: 0.1, 2000: 0.01, 8000: 0.0025},
    "DGTSK": {4: 0.2, 50: 0.2, 2000: 0.005},
    "AYATSK": {4: 0.01, 1000: 0.01, 1001: 0.001},
    "ADPTSK": {4: 0.01, 1000: 0.01, 1001: 0.001},
    "ADMTSK": {4: 0.01, 1000: 0.01, 1001: 0.001},
    "DombiTSK": {4: 0.01, 1000: 0.01, 1001: 0.001},
}


def _data(n_features: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((40, n_features)).astype(np.float32)
    return x, (x[:, 0] > 0.5).astype(int)


@pytest.mark.parametrize("family", sorted(RULES))
@pytest.mark.parametrize("kind", ["Classifier", "Regressor"])
def test_rule_of_each_family(family: str, kind: str) -> None:
    estimator = getattr(highfis, family + kind)()
    assert estimator.get_params()["learning_rate"] == "auto"
    for n_features, expected in RULES[family].items():
        assert estimator._resolve_learning_rate(n_features) == pytest.approx(expected)


@pytest.mark.parametrize("family", ["ADATSK", "AYATSK", "ADPTSK"])
def test_fit_records_the_resolved_value(family: str) -> None:
    x, y = _data()
    model = getattr(highfis, family + "Classifier")(epochs=2, random_state=0).fit(x, y)
    assert model.learning_rate_ == RULES[family][4]
    assert model.history_["config"]["learning_rate"] == pytest.approx(RULES[family][4])


def test_a_number_always_wins() -> None:
    x, y = _data()
    model = highfis.ADATSKClassifier(learning_rate=0.03, epochs=2, random_state=0).fit(x, y)
    assert model.learning_rate_ == 0.03


def test_auto_is_refused_where_no_rule_exists() -> None:
    x, y = _data()
    with pytest.raises(ValueError, match="learning_rate='auto' is not available"):
        arguments: dict[str, Any] = {"learning_rate": "auto", "epochs": 1}
        highfis.HTSKClassifier(**arguments).fit(x, y)


@pytest.mark.parametrize("family", ["ADATSK", "AYATSK", "ADPTSK", "DGTSK"])
def test_auto_survives_clone_and_persistence(family: str, tmp_path: Path) -> None:
    x, y = _data()
    kwargs: dict[str, Any] = {"dg_epochs": 3, "finetune_epochs": 3} if family == "DGTSK" else {"epochs": 3}
    model = getattr(highfis, family + "Classifier")(random_state=0, **kwargs).fit(x, y)
    assert clone(model).get_params()["learning_rate"] == "auto"
    path = str(tmp_path / "model.pt")
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.get_params()["learning_rate"] == "auto"
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
