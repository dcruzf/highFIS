from __future__ import annotations

import numpy as np

from highfis import AYATSKClassifier, AYATSKRegressor


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return (x, y)


def test_ayatsk_classifier_estimator_fit_predict_score() -> None:
    x, y = _make_dataset(80)
    est = AYATSKClassifier(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    proba = est.predict_proba(x)
    pred = est.predict(x)
    score = est.score(x, y)
    assert proba.shape == (x.shape[0], 2)
    assert pred.shape == (x.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-06)
    assert 0.0 <= score <= 1.0


def test_ayatsk_classifier_estimator_fit_predict_score_short() -> None:
    x, y = _make_dataset(80)
    est = AYATSKClassifier(n_mfs=2, mf_init="kmeans", epochs=3, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    proba = est.predict_proba(x)
    pred = est.predict(x)
    score = est.score(x, y)
    assert proba.shape == (x.shape[0], 2)
    assert pred.shape == (x.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-06)
    assert 0.0 <= score <= 1.0


def test_ayatsk_regressor_estimator_fit_predict() -> None:
    x = np.random.default_rng(123).normal(size=(40, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1]
    est = AYATSKRegressor(n_mfs=2, mf_init="kmeans", epochs=3, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)
