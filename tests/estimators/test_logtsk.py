from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from highfis.defuzzifiers import InvLogDefuzzifier
from highfis.estimators import LogTSKClassifier, LogTSKRegressor


def _make_clf_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return (x, y)


def _make_reg_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(456)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2]).astype(np.float32)
    return (x, y)


class TestLogTSKClassifierEstimator:
    def test_fit_predict_score_kmeans(self) -> None:
        x, y = _make_clf_dataset(80)
        est = LogTSKClassifier(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
        est.fit(x, y)
        proba = est.predict_proba(x)
        pred = est.predict(x)
        score = est.score(x, y)
        assert proba.shape == (x.shape[0], 2)
        assert pred.shape == (x.shape[0],)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-06)
        assert 0.0 <= score <= 1.0

    def test_fit_predict_grid(self) -> None:
        x, y = _make_clf_dataset(80)
        est = LogTSKClassifier(n_mfs=2, mf_init="grid", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
        est.fit(x, y)
        pred = est.predict(x)
        assert pred.shape == (x.shape[0],)

    def test_model_uses_log_sum_defuzzifier(self) -> None:
        x, y = _make_clf_dataset(40)
        est = LogTSKClassifier(n_mfs=2, epochs=1, batch_size=16, random_state=0)
        est.fit(x, y)
        assert isinstance(est.model_.defuzzifier, InvLogDefuzzifier)

    def test_predict_requires_fit(self) -> None:
        x, _ = _make_clf_dataset(10)
        est = LogTSKClassifier(n_mfs=2, epochs=1, batch_size=16)
        with pytest.raises(NotFittedError):
            est.predict_proba(x)


class TestLogTSKRegressorEstimator:
    def test_fit_predict_score_kmeans(self) -> None:
        x, y = _make_reg_dataset(80)
        est = LogTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
        est.fit(x, y)
        pred = est.predict(x)
        assert pred.shape == (x.shape[0],)
        r2 = est.score(x, y)
        assert isinstance(r2, float)

    def test_fit_predict_grid(self) -> None:
        x, y = _make_reg_dataset(80)
        est = LogTSKRegressor(n_mfs=2, mf_init="grid", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
        est.fit(x, y)
        pred = est.predict(x)
        assert pred.shape == (x.shape[0],)

    def test_model_uses_log_sum_defuzzifier(self) -> None:
        x, y = _make_reg_dataset(40)
        est = LogTSKRegressor(n_mfs=2, epochs=1, batch_size=16, random_state=0)
        est.fit(x, y)
        assert isinstance(est.model_.defuzzifier, InvLogDefuzzifier)

    def test_predict_requires_fit(self) -> None:
        x, _ = _make_reg_dataset(10)
        est = LogTSKRegressor(n_mfs=2, epochs=1, batch_size=16)
        with pytest.raises(NotFittedError):
            est.predict(x)
