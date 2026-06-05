from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from highfis.defuzzifiers import SumBasedDefuzzifier
from highfis.estimators import InputConfig, TSKClassifier, TSKRegressor
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


class TestTSKClassifierEstimator:
    def test_fit_predict_score_kmeans(self) -> None:
        x, y = _make_clf_dataset(80)
        est = TSKClassifier(
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

    def test_fit_predict_score_fcm(self) -> None:
        x, y = _make_clf_dataset(80)
        est = TSKClassifier(
            n_mfs=2,
            mf_init="fcm",
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
        est = TSKClassifier(
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
        est = TSKClassifier(n_mfs=2, epochs=1, batch_size=16, random_state=0)
        est.fit(x, y)
        assert isinstance(est.model_.defuzzifier, SumBasedDefuzzifier)

    def test_predict_requires_fit(self) -> None:
        x, _ = _make_clf_dataset(10)
        est = TSKClassifier(n_mfs=2, epochs=1, batch_size=16)
        with pytest.raises(NotFittedError):
            est.predict_proba(x)

    def test_validates_input_config_length(self) -> None:
        x, y = _make_clf_dataset(20)
        est = TSKClassifier(
            input_configs=[InputConfig(name="x1", n_mfs=2)],
            batch_size=16,
        )
        with pytest.raises(ValueError, match="input_configs length"):
            est.fit(x, y)


class TestTSKRegressorEstimator:
    def test_fit_predict_score_kmeans(self) -> None:
        x, y = _make_reg_dataset(80)
        est = TSKRegressor(
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

    def test_fit_predict_score_fcm(self) -> None:
        x, y = _make_reg_dataset(80)
        est = TSKRegressor(
            n_mfs=2,
            mf_init="fcm",
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
        est = TSKRegressor(
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
        est = TSKRegressor(n_mfs=2, epochs=1, batch_size=16, random_state=0)
        est.fit(x, y)
        assert isinstance(est.model_.defuzzifier, SumBasedDefuzzifier)

    def test_predict_requires_fit(self) -> None:
        x, _ = _make_reg_dataset(10)
        est = TSKRegressor(n_mfs=2, epochs=1, batch_size=16)
        with pytest.raises(NotFittedError):
            est.predict(x)


def test_tsk_classifier_paper_strict_uses_paper_protocol_defaults() -> None:
    est = TSKClassifier(paper_strict=True)
    assert est.n_mfs == 30
    assert est.mf_init == "kmeans"
    assert est.sigma_scale == 1.0
    assert est.rule_base == "coco"
    assert est.epochs == 200
    assert est.learning_rate == 1e-2
    assert est.batch_size == 512


def test_tsk_regressor_paper_strict_uses_paper_protocol_defaults() -> None:
    est = TSKRegressor(paper_strict=True)
    assert est.n_mfs == 30
    assert est.mf_init == "kmeans"
    assert est.sigma_scale == 1.0
    assert est.rule_base == "coco"
    assert est.epochs == 200
    assert est.learning_rate == 1e-2
    assert est.batch_size == 512


def test_tsk_paper_strict_rejects_conflicting_hyperparameters() -> None:
    with pytest.raises(ValueError, match=r"paper_strict requires n_mfs=30"):
        TSKClassifier(paper_strict=True, n_mfs=3)
    with pytest.raises(ValueError, match=r"paper_strict requires mf_init='kmeans'"):
        TSKClassifier(paper_strict=True, mf_init="grid")
    with pytest.raises(ValueError, match=r"paper_strict requires sigma_scale=1.0"):
        TSKRegressor(paper_strict=True, sigma_scale=2.0)
    with pytest.raises(ValueError, match=r"paper_strict requires rule_base='coco'"):
        TSKRegressor(paper_strict=True, rule_base="cartesian")
    with pytest.raises(ValueError, match=r"paper_strict requires epochs=200"):
        TSKClassifier(paper_strict=True, epochs=10)
    with pytest.raises(ValueError, match=r"paper_strict requires learning_rate=1e-2"):
        TSKClassifier(paper_strict=True, learning_rate=1e-3)
    with pytest.raises(ValueError, match=r"paper_strict requires batch_size=512"):
        TSKRegressor(paper_strict=True, batch_size=256)


def test_tsk_paper_strict_trainers() -> None:
    x = np.random.randn(45, 2)
    y = np.random.randint(0, 2, (45,))

    # Test TSK (use default 200 epochs as locked by paper_strict)
    est_tsk = TSKClassifier(paper_strict=True)
    est_tsk.fit(x, y)
    assert isinstance(est_tsk._get_trainer(), _HTSKPaperStrictTrainer)
    assert est_tsk.history_["stopped_epoch"] is not None
