from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from highfis import FSREADATSKClassifier, FSREADATSKRegressor
from highfis.estimators._fsre import _validate_adatsk_paper_strict_input_range
from highfis.optim import FSRETrainer


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return x, y


def _make_regression_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(456)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2]).astype(np.float32)
    return x, y


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


def test_fsre_adatsk_classifier_paper_strict_defaults() -> None:
    clf = FSREADATSKClassifier(paper_strict=True)
    assert clf.n_mfs == 5
    assert clf.mf_init == "grid"
    assert clf.sigma_scale == 1.0
    assert clf.rule_base == "coco"
    assert clf.use_en_frb is True
    assert clf.learning_rate == 1e-2
    assert clf.batch_size is None
    assert clf.fs_epochs == 200
    assert clf.re_epochs == 200
    assert clf.finetune_epochs == 200


def test_fsre_adatsk_classifier_paper_strict_overrides_raise() -> None:
    with pytest.raises(ValueError, match="paper_strict requires n_mfs=5"):
        FSREADATSKClassifier(paper_strict=True, n_mfs=3)
    with pytest.raises(ValueError, match="paper_strict requires mf_init='grid'"):
        FSREADATSKClassifier(paper_strict=True, mf_init="fcm")
    with pytest.raises(ValueError, match=r"paper_strict requires sigma_scale=1\.0"):
        FSREADATSKClassifier(paper_strict=True, sigma_scale=0.5)
    with pytest.raises(ValueError, match="paper_strict requires rule_base='coco'"):
        FSREADATSKClassifier(paper_strict=True, rule_base="cartesian")
    with pytest.raises(ValueError, match=r"paper_strict requires learning_rate=1e-2"):
        FSREADATSKClassifier(paper_strict=True, learning_rate=1e-3)
    with pytest.raises(ValueError, match="paper_strict requires batch_size=None"):
        FSREADATSKClassifier(paper_strict=True, batch_size=128)
    with pytest.raises(ValueError, match="paper_strict requires fs_epochs=200"):
        FSREADATSKClassifier(paper_strict=True, fs_epochs=5)
    with pytest.raises(ValueError, match="paper_strict requires re_epochs=200"):
        FSREADATSKClassifier(paper_strict=True, re_epochs=5)
    with pytest.raises(ValueError, match="paper_strict requires finetune_epochs=200"):
        FSREADATSKClassifier(paper_strict=True, finetune_epochs=5)


def test_fsre_adatsk_classifier_paper_strict_low_dim_zeta_fit() -> None:
    clf = FSREADATSKClassifier(paper_strict=True, fs_epochs=1, re_epochs=1, finetune_epochs=1)
    x = np.random.default_rng(0).uniform(0, 1, size=(2, 5))
    y = np.array([0, 1])

    with patch("highfis.optim.FSRETrainer.fit", return_value={"train": [], "stopped_epoch": 0}):
        clf.fit(x, y)
    assert clf.zeta_lambda == 0.5
    assert clf.zeta_theta == 0.3

    clf_bad = FSREADATSKClassifier(paper_strict=True, zeta_lambda=0.4, fs_epochs=1, re_epochs=1, finetune_epochs=1)
    with (
        pytest.raises(ValueError, match=r"paper_strict requires zeta_lambda=0\.5 for low-dimensional data"),
        patch("highfis.optim.FSRETrainer.fit", return_value={"train": [], "stopped_epoch": 0}),
    ):
        clf_bad.fit(x, y)


def test_fsre_adatsk_classifier_paper_strict_high_dim_zeta_fit() -> None:
    clf = FSREADATSKClassifier(paper_strict=True, fs_epochs=1, re_epochs=1, finetune_epochs=1)
    x = np.random.default_rng(0).uniform(0, 1, size=(2, 1000))
    y = np.array([0, 1])

    with patch("highfis.optim.FSRETrainer.fit", return_value={"train": [], "stopped_epoch": 0}):
        clf.fit(x, y)
    assert clf.zeta_lambda == 0.4
    assert clf.zeta_theta == 0.5

    clf_bad_explicit = FSREADATSKClassifier(
        paper_strict=True, zeta_lambda=0.9, fs_epochs=1, re_epochs=1, finetune_epochs=1
    )
    with (
        pytest.raises(ValueError, match=r"paper_strict requires zeta_lambda=0\.4 for high-dimensional data"),
        patch("highfis.optim.FSRETrainer.fit", return_value={"train": [], "stopped_epoch": 0}),
    ):
        clf_bad_explicit.fit(x, y)


def test_fsre_adatsk_classifier_paper_strict_input_range() -> None:
    clf = FSREADATSKClassifier(paper_strict=True, fs_epochs=1, re_epochs=1, finetune_epochs=1)
    x_bad = np.array([[-0.1, 0.5], [1.1, 0.5]])
    y = np.array([0, 1])

    with (
        pytest.raises(ValueError, match="paper_strict requires x to be linearly normalized to"),
        patch("highfis.optim.FSRETrainer.fit", return_value={"train": [], "stopped_epoch": 0}),
    ):
        clf.fit(x_bad, y)


def test_fsre_adatsk_regressor_no_paper_strict_support() -> None:
    with pytest.raises(TypeError):
        FSREADATSKRegressor(paper_strict=True)  # type: ignore


# --- Additional Estimator tests from test_estimators.py ---


def test_fsre_adatsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = FSREADATSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=1.5,
        fs_epochs=5,
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


def test_fsre_adatsk_classifier_estimator_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_init must be > 0"):
        FSREADATSKClassifier(n_mfs=2, mf_init="kmeans", lambda_init=0.0, fs_epochs=1, batch_size=16)


def test_fsre_adatsk_regressor_estimator_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_init must be > 0"):
        FSREADATSKRegressor(n_mfs=2, mf_init="kmeans", lambda_init=0.0, fs_epochs=1, batch_size=16)


def test_fsre_adatsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_regression_dataset(80)
    est = FSREADATSKRegressor(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=2.0,
        fs_epochs=5,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (x.shape[0],)


def test_paper_strict_input_range_empty_arrays_fsre() -> None:
    empty = np.array([])
    _validate_adatsk_paper_strict_input_range(empty)


def test_fit_with_strict_and_validation_data_fsre() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, (40, 2)).astype(np.float32)
    y = rng.choice([0, 1], size=40).astype(np.int64)
    x_val = rng.uniform(0.0, 1.0, (10, 2)).astype(np.float32)
    y_val = rng.choice([0, 1], size=10).astype(np.int64)

    clf_fsre = FSREADATSKClassifier(paper_strict=True, fs_epochs=200, re_epochs=200, finetune_epochs=200)
    clf_fsre.fit(x, y, x_val=x_val, y_val=y_val)


def test_fsre_strict_zeta_theta_validation() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, (5, 2)).astype(np.float32)
    y = rng.choice([0, 1], size=5).astype(np.int64)

    clf_low = FSREADATSKClassifier(
        paper_strict=True,
        zeta_theta=0.5,
        fs_epochs=200,
        re_epochs=200,
        finetune_epochs=200,
    )
    with pytest.raises(ValueError, match=r"paper_strict requires zeta_theta=0\.3 for low-dimensional data"):
        clf_low.fit(x, y)

    x_high = rng.uniform(0.0, 1.0, (5, 1005)).astype(np.float32)
    y_high = rng.choice([0, 1], size=5).astype(np.int64)

    clf = FSREADATSKClassifier(
        paper_strict=True,
        zeta_lambda=0.4,
        zeta_theta=0.3,
        fs_epochs=200,
        re_epochs=200,
        finetune_epochs=200,
    )
    with pytest.raises(ValueError, match=r"paper_strict requires zeta_theta=0\.5 for high-dimensional data"):
        clf.fit(x_high, y_high)


def test_fsre_epochs_strict_validation() -> None:
    with pytest.raises(ValueError, match="paper_strict requires fs_epochs=200"):
        FSREADATSKClassifier(paper_strict=True, fs_epochs=5)


def test_fsre_invalid_en_frb_strict_validation() -> None:
    with pytest.raises(ValueError, match="paper_strict requires use_en_frb=True"):
        FSREADATSKClassifier(paper_strict=True, use_en_frb="invalid")  # type: ignore
