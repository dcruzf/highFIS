from __future__ import annotations

import numpy as np

from highfis import DombiTSKClassifier, DombiTSKRegressor


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


def test_dombi_tsk_classifier_estimator_fit_predict() -> None:
    x, y = _make_dataset(60)
    est = DombiTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_dombi_tsk_regressor_estimator_fit_predict_score() -> None:
    x, y = _make_regression_dataset(80)
    est = DombiTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)
