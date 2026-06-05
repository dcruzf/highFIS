from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from highfis import AYATSKClassifier, AYATSKRegressor


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return x, y


def test_ayatsk_classifier_estimator_fit_predict_score() -> None:
    x, y = _make_dataset(80)
    est = AYATSKClassifier(
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


def test_ayatsk_classifier_estimator_fit_predict_score_short() -> None:
    x, y = _make_dataset(80)
    est = AYATSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
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


def test_ayatsk_regressor_estimator_fit_predict() -> None:
    x = np.random.default_rng(123).normal(size=(40, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1]
    est = AYATSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (x.shape[0],)


def test_ayatsk_classifier_default_setup_is_paper_style() -> None:
    x, y = _make_dataset(80)
    est = AYATSKClassifier(epochs=1, random_state=7)
    est.fit(x, y)

    assert est.rule_base_ == "coco"
    assert est.model_.n_rules == 3
    assert est.model_._default_criterion().__class__.__name__ == "MSELoss"
    assert est._resolve_default_batch_size(80) is None
    assert est._resolve_default_batch_size(500) == 50

    mfs = est.model_.input_mfs["x1"]
    assert len(mfs) == 3
    assert type(mfs[0]).__name__ == "CompositeExponentialMF"
    assert float(cast(Any, est.model_).lambda_) > 0.0


def test_ayatsk_classifier_paper_strict_uses_paper_protocol_defaults() -> None:
    est = AYATSKClassifier(paper_strict=True)

    assert est.n_mfs == 3
    assert est.mf_init == "grid"
    assert est.sigma_scale == 1.0
    assert est.rule_base == "coco"
    assert est.epochs == 200
    assert est.learning_rate == 1e-3
    assert est.k == 10.0
    assert est.paper_strict is True


def test_ayatsk_classifier_paper_strict_rejects_conflicting_hyperparameters() -> None:
    with pytest.raises(ValueError, match="paper_strict requires n_mfs=3"):
        AYATSKClassifier(paper_strict=True, n_mfs=4)
    with pytest.raises(ValueError, match="paper_strict requires mf_init='grid'"):
        AYATSKClassifier(paper_strict=True, mf_init="kmeans")
    with pytest.raises(ValueError, match=r"paper_strict requires sigma_scale=1\.0"):
        AYATSKClassifier(paper_strict=True, sigma_scale=0.8)
    with pytest.raises(ValueError, match="paper_strict requires rule_base='coco'"):
        AYATSKClassifier(paper_strict=True, rule_base="cartesian")
    with pytest.raises(ValueError, match="paper_strict requires epochs=200"):
        AYATSKClassifier(paper_strict=True, epochs=100)
    with pytest.raises(ValueError, match="paper_strict requires learning_rate=1e-3"):
        AYATSKClassifier(paper_strict=True, learning_rate=1e-2)


def test_ayatsk_estimators_validate_k_parameter() -> None:
    with pytest.raises(ValueError, match=r"k must be > 1\.0"):
        AYATSKClassifier(k=1.0)
    with pytest.raises(ValueError, match=r"k must be > 1\.0"):
        AYATSKRegressor(k=1.0)


def test_ayatsk_classifier_paper_strict_warns_batch_policy_low_dimensional_case() -> None:
    x = np.zeros((600, 3), dtype=np.float32)
    y = np.zeros((599,), dtype=np.int64)
    est = AYATSKClassifier(paper_strict=True)

    with pytest.warns(UserWarning, match="paper_strict: mini-batch policy may diverge"), pytest.raises(ValueError):
        est.fit(x, y)


def test_ayatsk_regressor_default_setup_is_paper_style() -> None:
    x = np.random.default_rng(123).normal(size=(40, 3)).astype(np.float32)
    y = x[:, 0] + 0.5 * x[:, 1]
    est = AYATSKRegressor(epochs=1, random_state=7)
    est.fit(x, y)

    assert est.rule_base_ == "coco"
    assert est.model_._default_criterion().__class__.__name__ == "MSELoss"
    assert est._resolve_default_batch_size(80) is None
    assert est._resolve_default_batch_size(500) == 50
    assert float(cast(Any, est.model_).lambda_) > 0.0


def test_yager_strict_rejects_small_k() -> None:
    with pytest.raises(ValueError, match=r"paper_strict requires k > 1\.0"):
        AYATSKClassifier(paper_strict=True, k=1.0)
