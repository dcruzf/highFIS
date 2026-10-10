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


def test_fsre_adatsk_defaults_enable_consequent_batch_norm() -> None:
    """FSRE-ADATSK enables consequent batch norm by default (guards the collapse fix).

    The first-order gated consequent spans all features and is trained with plain
    gradient descent, which diverges (weights -> NaN) on high-dimensional data
    without normalisation, collapsing the model to a single class (see
    investigate_gating_collapse.py). Batch norm on the consequent inputs is the
    default, mirroring the ADATSK fix. Validated end-to-end on the real Colon
    dataset (D=2000): with the default the classifier predicts both classes again.
    """
    assert FSREADATSKClassifier().consequent_batch_norm is True
    assert FSREADATSKRegressor().consequent_batch_norm is True

    # After a small fit the batch-norm module must be active on the model.
    x, y = _make_dataset(60)
    clf = FSREADATSKClassifier(n_mfs=2, mf_init="kmeans", fs_epochs=1, re_epochs=1, finetune_epochs=1, random_state=7)
    clf.fit(x, y)
    assert clf.model_.consequent_bn is not None


def test_fsre_adatsk_regressor_estimator_default_uses_fsre_trainer() -> None:
    reg = FSREADATSKRegressor()
    assert isinstance(reg._get_trainer(), FSRETrainer)


def test_fsre_adatsk_classifier_predict_proba_wrong_n_features() -> None:
    X = np.random.default_rng(0).standard_normal((20, 3))
    y = np.random.default_rng(0).integers(0, 2, size=20)
    clf = FSREADATSKClassifier(fs_epochs=1, re_epochs=1, finetune_epochs=1)
    clf.fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict_proba(X[:, :2])


def test_fsre_adatsk_regressor_predict_wrong_n_features() -> None:
    X = np.random.default_rng(1).standard_normal((20, 3))
    y = np.random.default_rng(1).standard_normal(20)
    reg = FSREADATSKRegressor(fs_epochs=1, re_epochs=1, finetune_epochs=1)
    reg.fit(X, y)
    with pytest.raises(ValueError, match="features"):
        reg.predict(X[:, :2])


def test_fsre_adatsk_classifier_estimator_fit_predict_proba_predict_score() -> None:
    x, y = _make_dataset(80)
    est = FSREADATSKClassifier(
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=1.5,
        fs_epochs=5,
        re_epochs=5,
        finetune_epochs=5,
        learning_rate=0.01,
        random_state=7,
        batch_size=16,
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
        n_mfs=2,
        mf_init="kmeans",
        lambda_init=2.0,
        fs_epochs=5,
        re_epochs=5,
        finetune_epochs=5,
        learning_rate=0.01,
        random_state=7,
        batch_size=16,
    )
    est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)


@pytest.mark.parametrize("estimator_cls", [FSREADATSKClassifier, FSREADATSKRegressor])
def test_membership_function_of_the_article(estimator_cls: type) -> None:
    """FSRE-AdaTSK uses exp(-(x - m)^2): no spread, centres between the minimum and the maximum."""
    rng = np.random.default_rng(0)
    x = rng.random((80, 4)).astype(np.float32)
    x[0], x[1] = 0.0, 1.0
    y = (x[:, 0] > 0.5).astype(int) if estimator_cls is FSREADATSKClassifier else x[:, 0]
    model = estimator_cls(fs_epochs=20, re_epochs=20, finetune_epochs=20, random_state=0).fit(x, y)
    sets = [mf for group in model.get_mf_params().values() for mf in group]
    assert {mf["type"] for mf in sets} == {"ADATSKGaussianMF"}
    assert [mf["sigma"] for mf in sets] == pytest.approx([1.0] * len(sets), abs=1e-5)
    trainable = [name for name, param in model.model_.named_parameters() if param.requires_grad]
    assert "membership_layer._flat_raw_sigma" not in trainable
