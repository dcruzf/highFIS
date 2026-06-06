import numpy as np
import pytest

from highfis import ADMTSKClassifier, ADMTSKRegressor, DombiTSKClassifier, DombiTSKRegressor


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
    assert est.__sklearn_is_fitted__() is True


def test_dombi_tsk_regressor_estimator_fit_predict_score() -> None:
    x, y = _make_regression_dataset(80)
    est = DombiTSKRegressor(n_mfs=2, mf_init="kmeans", epochs=5, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


def test_admtsk_estimators() -> None:
    x, y = _make_dataset(40)
    # Test ADMTSKClassifier default profile
    clf = ADMTSKClassifier(epochs=1)
    clf.fit(x, y)
    assert clf.__sklearn_is_fitted__() is True

    # Test feature count error
    x_1d = x[:, :1]
    with pytest.raises(ValueError, match="requires at least 2 features"):
        clf.fit(x_1d, y)

    # Test ADMTSKRegressor default profile
    x_reg, y_reg = _make_regression_dataset(40)
    reg = ADMTSKRegressor(epochs=1)
    reg.fit(x_reg, y_reg)
    assert reg.__sklearn_is_fitted__() is True

    # Test feature count error for regressor
    with pytest.raises(ValueError, match="requires at least 2 features"):
        reg.fit(x_1d, y_reg)


def test_dombi_default_profiles() -> None:
    x, y = _make_dataset(40)
    # n_mfs=3 and mf_init="grid"
    clf = DombiTSKClassifier(n_mfs=3, mf_init="grid", epochs=1)
    clf.fit(x, y)
    assert clf.model_ is not None

    x_reg, y_reg = _make_regression_dataset(40)
    reg = DombiTSKRegressor(n_mfs=3, mf_init="grid", epochs=1)
    reg.fit(x_reg, y_reg)
    assert reg.model_ is not None
