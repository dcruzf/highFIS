"""Tests for TSK and LogTSK sklearn-compatible estimators.

Verifies the fit/predict/score contract, both MF-initialization modes
(kmeans and grid), and parameter passthrough for all 4 new estimators.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from highfis.defuzzifiers import LogSumDefuzzifier, SumBasedDefuzzifier
from highfis.estimators import (
    InputConfig,
    LogTSKClassifierEstimator,
    LogTSKRegressorEstimator,
    TSKClassifierEstimator,
    TSKRegressorEstimator,
)


def _make_clf_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return x, y


def _make_reg_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(456)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2]).astype(np.float32)
    return x, y


# =====================================================================
# TSKClassifierEstimator
# =====================================================================


class TestTSKClassifierEstimator:
    def test_fit_predict_score_kmeans(self) -> None:
        x, y = _make_clf_dataset(80)
        est = TSKClassifierEstimator(
            n_mfs=2,
            mf_init="kmeans",
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

    def test_fit_predict_grid(self) -> None:
        x, y = _make_clf_dataset(80)
        est = TSKClassifierEstimator(
            n_mfs=2,
            mf_init="grid",
            epochs=5,
            learning_rate=1e-2,
            random_state=7,
            batch_size=16,
        )
        est.fit(x, y)
        pred = est.predict(x)
        assert pred.shape == (x.shape[0],)

    def test_model_uses_sum_based_defuzzifier(self) -> None:
        x, y = _make_clf_dataset(40)
        est = TSKClassifierEstimator(n_mfs=2, epochs=1, batch_size=16, random_state=0)
        est.fit(x, y)
        assert isinstance(est.model_.defuzzifier, SumBasedDefuzzifier)

    def test_predict_requires_fit(self) -> None:
        x, _ = _make_clf_dataset(10)
        est = TSKClassifierEstimator(n_mfs=2, epochs=1, batch_size=16)
        with pytest.raises(NotFittedError):
            est.predict_proba(x)

    def test_validates_input_config_length(self) -> None:
        x, y = _make_clf_dataset(20)
        est = TSKClassifierEstimator(
            input_configs=[InputConfig(name="x1", n_mfs=2)],
            batch_size=16,
        )
        with pytest.raises(ValueError, match="input_configs length"):
            est.fit(x, y)


# =====================================================================
# TSKRegressorEstimator
# =====================================================================


class TestTSKRegressorEstimator:
    def test_fit_predict_score_kmeans(self) -> None:
        x, y = _make_reg_dataset(80)
        est = TSKRegressorEstimator(
            n_mfs=2,
            mf_init="kmeans",
            epochs=5,
            learning_rate=1e-2,
            random_state=7,
            batch_size=16,
        )
        est.fit(x, y)
        pred = est.predict(x)
        assert pred.shape == (x.shape[0],)
        # score() returns R² — should be a float
        r2 = est.score(x, y)
        assert isinstance(r2, float)

    def test_fit_predict_grid(self) -> None:
        x, y = _make_reg_dataset(80)
        est = TSKRegressorEstimator(
            n_mfs=2,
            mf_init="grid",
            epochs=5,
            learning_rate=1e-2,
            random_state=7,
            batch_size=16,
        )
        est.fit(x, y)
        pred = est.predict(x)
        assert pred.shape == (x.shape[0],)

    def test_model_uses_sum_based_defuzzifier(self) -> None:
        x, y = _make_reg_dataset(40)
        est = TSKRegressorEstimator(n_mfs=2, epochs=1, batch_size=16, random_state=0)
        est.fit(x, y)
        assert isinstance(est.model_.defuzzifier, SumBasedDefuzzifier)

    def test_predict_requires_fit(self) -> None:
        x, _ = _make_reg_dataset(10)
        est = TSKRegressorEstimator(n_mfs=2, epochs=1, batch_size=16)
        with pytest.raises(NotFittedError):
            est.predict(x)


# =====================================================================
# LogTSKClassifierEstimator
# =====================================================================


class TestLogTSKClassifierEstimator:
    def test_fit_predict_score_kmeans(self) -> None:
        x, y = _make_clf_dataset(80)
        est = LogTSKClassifierEstimator(
            n_mfs=2,
            mf_init="kmeans",
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

    def test_fit_predict_grid(self) -> None:
        x, y = _make_clf_dataset(80)
        est = LogTSKClassifierEstimator(
            n_mfs=2,
            mf_init="grid",
            epochs=5,
            learning_rate=1e-2,
            random_state=7,
            batch_size=16,
        )
        est.fit(x, y)
        pred = est.predict(x)
        assert pred.shape == (x.shape[0],)

    def test_model_uses_log_sum_defuzzifier(self) -> None:
        x, y = _make_clf_dataset(40)
        est = LogTSKClassifierEstimator(n_mfs=2, epochs=1, batch_size=16, random_state=0)
        est.fit(x, y)
        assert isinstance(est.model_.defuzzifier, LogSumDefuzzifier)

    def test_predict_requires_fit(self) -> None:
        x, _ = _make_clf_dataset(10)
        est = LogTSKClassifierEstimator(n_mfs=2, epochs=1, batch_size=16)
        with pytest.raises(NotFittedError):
            est.predict_proba(x)


# =====================================================================
# LogTSKRegressorEstimator
# =====================================================================


class TestLogTSKRegressorEstimator:
    def test_fit_predict_score_kmeans(self) -> None:
        x, y = _make_reg_dataset(80)
        est = LogTSKRegressorEstimator(
            n_mfs=2,
            mf_init="kmeans",
            epochs=5,
            learning_rate=1e-2,
            random_state=7,
            batch_size=16,
        )
        est.fit(x, y)
        pred = est.predict(x)
        assert pred.shape == (x.shape[0],)
        r2 = est.score(x, y)
        assert isinstance(r2, float)

    def test_fit_predict_grid(self) -> None:
        x, y = _make_reg_dataset(80)
        est = LogTSKRegressorEstimator(
            n_mfs=2,
            mf_init="grid",
            epochs=5,
            learning_rate=1e-2,
            random_state=7,
            batch_size=16,
        )
        est.fit(x, y)
        pred = est.predict(x)
        assert pred.shape == (x.shape[0],)

    def test_model_uses_log_sum_defuzzifier(self) -> None:
        x, y = _make_reg_dataset(40)
        est = LogTSKRegressorEstimator(n_mfs=2, epochs=1, batch_size=16, random_state=0)
        est.fit(x, y)
        assert isinstance(est.model_.defuzzifier, LogSumDefuzzifier)

    def test_predict_requires_fit(self) -> None:
        x, _ = _make_reg_dataset(10)
        est = LogTSKRegressorEstimator(n_mfs=2, epochs=1, batch_size=16)
        with pytest.raises(NotFittedError):
            est.predict(x)
