from __future__ import annotations

import numpy as np
import pytest

from highfis import FSREADATSKClassifier, FSREADATSKRegressor
from highfis.optim import FSRETrainer


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


def test_fsre_adatsk_classifier_estimator_default_uses_fsre_trainer() -> None:
    clf = FSREADATSKClassifier()
    assert isinstance(clf._get_trainer(), FSRETrainer)


def test_fsre_adatsk_regressor_estimator_default_uses_fsre_trainer() -> None:
    reg = FSREADATSKRegressor()
    assert isinstance(reg._get_trainer(), FSRETrainer)


def test_fsre_adatsk_classifier_predict_proba_wrong_n_features() -> None:
    X = np.random.default_rng(0).standard_normal((20, 3))
    y = np.random.default_rng(0).integers(0, 2, size=20)
    clf = FSREADATSKClassifier(fs_epochs=1, re_epochs=1, finetune_epochs=1)
    clf.fit(X, y)
    with pytest.raises(ValueError, match="expected"):
        clf.predict_proba(X[:, :2])


def test_fsre_adatsk_regressor_predict_wrong_n_features() -> None:
    X = np.random.default_rng(1).standard_normal((20, 3))
    y = np.random.default_rng(1).standard_normal(20)
    reg = FSREADATSKRegressor(fs_epochs=1, re_epochs=1, finetune_epochs=1)
    reg.fit(X, y)
    with pytest.raises(ValueError, match="expected"):
        reg.predict(X[:, :2])


def test_fsre_adatsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = FSREADATSKClassifier(
        n_mfs=2, mf_init="kmeans", lambda_init=1.5, fs_epochs=5, learning_rate=0.01, random_state=7, batch_size=16
    )
    est.fit(x, y)
    proba = est.predict_proba(x)
    pred = est.predict(x)
    score = est.score(x, y)
    assert proba.shape == (x.shape[0], 2)
    assert pred.shape == (x.shape[0],)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-06)
    assert 0.0 <= score <= 1.0


def test_fsre_adatsk_classifier_estimator_rejects_nonpositive_lambda() -> None:
    x, y = _make_dataset(80)
    est = FSREADATSKClassifier(n_mfs=2, mf_init="kmeans", lambda_init=0.0, fs_epochs=1, batch_size=16)
    with pytest.raises(ValueError, match="lambda_init must be > 0"):
        est.fit(x, y)


def test_fsre_adatsk_regressor_estimator_rejects_nonpositive_lambda() -> None:
    x, y = _make_regression_dataset(80)
    est = FSREADATSKRegressor(n_mfs=2, mf_init="kmeans", lambda_init=0.0, fs_epochs=1, batch_size=16)
    with pytest.raises(ValueError, match="lambda_init must be > 0"):
        est.fit(x, y)


def test_fsre_adatsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = FSREADATSKRegressor(
        n_mfs=2, mf_init="kmeans", lambda_init=2.0, fs_epochs=5, learning_rate=0.01, random_state=7, batch_size=16
    )
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)
