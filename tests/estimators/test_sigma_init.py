"""The initial spread of clustered fuzzy sets: cluster spread or the constant of the HTSK article."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone

import highfis

FAMILIES = ["TSK", "HTSK", "LogTSK"]


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((150, 6)).astype(np.float32)
    return x, (x[:, 0] > 0).astype(int)


def _sigmas(model: Any) -> np.ndarray:
    return np.array([mf["sigma"] for sets in model.get_mf_params().values() for mf in sets])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ["Classifier", "Regressor"])
def test_constant_spreads_follow_the_article(family: str, kind: str) -> None:
    x, y = _data()
    target = y if kind == "Classifier" else x[:, 1]
    cls = getattr(highfis, family + kind)
    model = cls(n_mfs=10, epochs=1, random_state=0, sigma_init="constant").fit(x, target)
    sigmas = _sigmas(model)
    # 60 draws from N(1, 0.2); training for one epoch moves them very little.
    assert sigmas.mean() == pytest.approx(1.0, abs=0.1)
    assert sigmas.std() == pytest.approx(0.2, abs=0.08)


def test_sigma_scale_is_the_centre_of_the_constant_draw() -> None:
    x, y = _data()
    model = highfis.TSKClassifier(n_mfs=10, epochs=1, random_state=0, sigma_init="constant", sigma_scale=5.0).fit(x, y)
    assert _sigmas(model).mean() == pytest.approx(5.0, abs=0.1)


def test_the_default_follows_the_cluster() -> None:
    x, y = _data()
    default = highfis.HTSKClassifier(n_mfs=4, epochs=1, random_state=0).fit(x, y)
    cluster = highfis.HTSKClassifier(n_mfs=4, epochs=1, random_state=0, sigma_init="cluster").fit(x, y)
    constant = highfis.HTSKClassifier(n_mfs=4, epochs=1, random_state=0, sigma_init="constant").fit(x, y)
    np.testing.assert_array_equal(_sigmas(default), _sigmas(cluster))
    assert not np.allclose(_sigmas(default), _sigmas(constant))


def test_an_unknown_value_is_refused() -> None:
    x, y = _data()
    with pytest.raises(ValueError, match="sigma_init must be one of"):
        highfis.HTSKClassifier(sigma_init="article").fit(x, y)


def test_the_choice_survives_clone_and_persistence(tmp_path: Path) -> None:
    x, y = _data()
    model = highfis.LogTSKClassifier(n_mfs=4, epochs=2, random_state=0, sigma_init="constant").fit(x, y)
    assert clone(model).get_params()["sigma_init"] == "constant"
    path = str(tmp_path / "model.pt")
    model.save(path)
    loaded = highfis.LogTSKClassifier.load(path)
    assert loaded.get_params()["sigma_init"] == "constant"
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
