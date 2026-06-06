import numpy as np
import pytest

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


def test_ayatsk_default_profile_and_errors() -> None:
    # Test classifier default profile (grid init, n_mfs=3, default batch size)
    x, y = _make_dataset(510)
    clf = AYATSKClassifier(epochs=1)
    clf.fit(x, y)
    assert clf.model_ is not None

    # Test classifier requires >= 2 features
    x_1d = x[:, :1]
    with pytest.raises(ValueError, match="requires at least 2 features"):
        clf.fit(x_1d, y)

    # Test classifier k must be > 1.0
    clf_bad = AYATSKClassifier(k=0.5, epochs=1)
    with pytest.raises(ValueError, match=r"k must be > 1.0"):
        clf_bad.fit(x, y)

    # Test regressor default profile (grid init, n_mfs=3, default batch size)
    x_reg = np.random.default_rng(123).normal(size=(510, 3)).astype(np.float32)
    y_reg = x_reg[:, 0] + 0.5 * x_reg[:, 1]
    reg = AYATSKRegressor(epochs=1)
    reg.fit(x_reg, y_reg)
    assert reg.model_ is not None

    # Test regressor requires >= 2 features
    with pytest.raises(ValueError, match="requires at least 2 features"):
        reg.fit(x_1d, y_reg)

    # Test regressor k must be > 1.0
    reg_bad = AYATSKRegressor(k=0.5, epochs=1)
    with pytest.raises(ValueError, match=r"k must be > 1.0"):
        reg_bad.fit(x_reg, y_reg)

    # Test batch size resolution for small dataset (< 500 samples)
    x_small, y_small = _make_dataset(40)
    clf_small = AYATSKClassifier(epochs=1)
    clf_small.fit(x_small, y_small)
    assert clf_small.batch_size is None
