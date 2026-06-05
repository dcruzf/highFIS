from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from highfis.defuzzifiers import InvLogDefuzzifier
from highfis.estimators import LogTSKClassifier, LogTSKRegressor
from highfis.estimators._htsk import _HTSKPaperStrictTrainer


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


class TestLogTSKClassifierEstimator:
    def test_fit_predict_score_kmeans(self) -> None:
        x, y = _make_clf_dataset(80)
        est = LogTSKClassifier(
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
        est = LogTSKClassifier(
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
        est = LogTSKRegressor(
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
        est = LogTSKRegressor(
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
        est = LogTSKRegressor(n_mfs=2, epochs=1, batch_size=16, random_state=0)
        est.fit(x, y)
        assert isinstance(est.model_.defuzzifier, InvLogDefuzzifier)

    def test_predict_requires_fit(self) -> None:
        x, _ = _make_reg_dataset(10)
        est = LogTSKRegressor(n_mfs=2, epochs=1, batch_size=16)
        with pytest.raises(NotFittedError):
            est.predict(x)


def test_logtsk_classifier_paper_strict_uses_paper_protocol_defaults() -> None:
    est = LogTSKClassifier(paper_strict=True)
    assert est.n_mfs == 30
    assert est.mf_init == "kmeans"
    assert est.sigma_scale == 1.0
    assert est.rule_base == "coco"
    assert est.epochs == 200
    assert est.learning_rate == 1e-2
    assert est.batch_size == 512


def test_logtsk_regressor_paper_strict_uses_paper_protocol_defaults() -> None:
    est = LogTSKRegressor(paper_strict=True)
    assert est.n_mfs == 30
    assert est.mf_init == "kmeans"
    assert est.sigma_scale == 1.0
    assert est.rule_base == "coco"
    assert est.epochs == 200
    assert est.learning_rate == 1e-2
    assert est.batch_size == 512


def test_logtsk_paper_strict_rejects_conflicting_hyperparameters() -> None:
    with pytest.raises(ValueError, match=r"paper_strict requires n_mfs=30"):
        LogTSKClassifier(paper_strict=True, n_mfs=3)
    with pytest.raises(ValueError, match=r"paper_strict requires mf_init='kmeans'"):
        LogTSKClassifier(paper_strict=True, mf_init="grid")
    with pytest.raises(ValueError, match=r"paper_strict requires sigma_scale=1.0"):
        LogTSKRegressor(paper_strict=True, sigma_scale=2.0)
    with pytest.raises(ValueError, match=r"paper_strict requires rule_base='coco'"):
        LogTSKRegressor(paper_strict=True, rule_base="cartesian")
    with pytest.raises(ValueError, match=r"paper_strict requires epochs=200"):
        LogTSKClassifier(paper_strict=True, epochs=10)
    with pytest.raises(ValueError, match=r"paper_strict requires learning_rate=1e-2"):
        LogTSKClassifier(paper_strict=True, learning_rate=1e-3)
    with pytest.raises(ValueError, match=r"paper_strict requires batch_size=512"):
        LogTSKRegressor(paper_strict=True, batch_size=256)


def test_logtsk_paper_strict_trainers() -> None:
    x = np.random.randn(45, 2)
    y = np.random.randint(0, 2, (45,))

    # Test LogTSK (use default 200 epochs as locked by paper_strict)
    est_log = LogTSKClassifier(paper_strict=True)
    est_log.fit(x, y)
    assert isinstance(est_log._get_trainer(), _HTSKPaperStrictTrainer)
    assert est_log.history_["stopped_epoch"] is not None


def test_logtsk_regressor_strict_trainer() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((31, 2)).astype(np.float32)
    y = rng.standard_normal((31,)).astype(np.float32)

    reg_log = LogTSKRegressor(paper_strict=True)
    reg_log.fit(x, y)
