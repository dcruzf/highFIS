from __future__ import annotations

import numpy as np

from highfis import HDFISMinClassifier, HDFISMinRegressor, HDFISProdClassifier, HDFISProdRegressor
from highfis.memberships import DimensionDependentGaussianMF
from highfis.models import HDFISMinClassifierModel, HDFISMinRegressorModel


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return (x, y)


def test_hdfis_prod_classifier_estimator_uses_dimension_dependent_mfs() -> None:
    x, y = _make_dataset(40)
    est = HDFISProdClassifier(n_mfs=2, mf_init="kmeans", epochs=3, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    assert all(isinstance(mf, DimensionDependentGaussianMF) for mfs in est.model_.input_mfs.values() for mf in mfs)


def test_hdfis_prod_regressor_estimator_uses_dimension_dependent_mfs() -> None:
    x = np.random.RandomState(0).normal(size=(40, 3)).astype(np.float32)
    y = (x[:, 0] * 0.5 + x[:, 1] * -0.2).astype(np.float32)
    est = HDFISProdRegressor(n_mfs=2, mf_init="kmeans", epochs=3, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    assert all(isinstance(mf, DimensionDependentGaussianMF) for mfs in est.model_.input_mfs.values() for mf in mfs)


def test_hdfis_min_classifier_estimator_builds_hdfis_min_model() -> None:
    x, y = _make_dataset(40)
    est = HDFISMinClassifier(n_mfs=2, mf_init="kmeans", epochs=2, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    assert isinstance(est.model_, HDFISMinClassifierModel)
    assert all(not p.requires_grad for p in est.model_.membership_layer.parameters())


def test_hdfis_min_regressor_estimator_builds_hdfis_min_model() -> None:
    x = np.random.RandomState(0).normal(size=(40, 3)).astype(np.float32)
    y = (x[:, 0] * 0.5 + x[:, 1] * -0.2).astype(np.float32)
    est = HDFISMinRegressor(n_mfs=2, mf_init="kmeans", epochs=2, learning_rate=0.01, random_state=7, batch_size=16)
    est.fit(x, y)
    assert isinstance(est.model_, HDFISMinRegressorModel)
    assert all(not p.requires_grad for p in est.model_.membership_layer.parameters())


def test_hdfis_feature_errors() -> None:
    import pytest

    x, y = _make_dataset(40)
    x_1d = x[:, :1]

    clf = HDFISProdClassifier(epochs=1)
    with pytest.raises(ValueError, match="requires at least 2 features"):
        clf.fit(x_1d, y)

    reg = HDFISProdRegressor(epochs=1)
    with pytest.raises(ValueError, match="requires at least 2 features"):
        reg.fit(x_1d, y)
