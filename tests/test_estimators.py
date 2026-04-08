from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from highfis.estimators import HTSKClassifierEstimator, InputConfig, _build_gaussian_input_mfs


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


def test_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = HTSKClassifierEstimator(
        n_mfs=2,
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


def test_estimator_predict_proba_requires_fit() -> None:
    x, _ = _make_dataset(10)
    est = HTSKClassifierEstimator(n_mfs=2, epochs=1)
    with pytest.raises(NotFittedError):
        est.predict_proba(x)


def test_estimator_validates_input_config_length() -> None:
    x, y = _make_dataset(20)
    est = HTSKClassifierEstimator(input_configs=[InputConfig(name="x1", n_mfs=2)])
    with pytest.raises(ValueError, match="input_configs length"):
        est.fit(x, y)
