from __future__ import annotations

import numpy as np
import pytest

from highfis import (
    HDFISMinClassifier,
    HDFISMinRegressor,
    HDFISProdClassifier,
    HDFISProdRegressor,
)
from highfis.memberships import DimensionDependentGaussianMF, GaussianMF
from highfis.models import HDFISMinClassifierModel, HDFISMinRegressorModel


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return x, y


def test_hdfis_prod_classifier_estimator_uses_dimension_dependent_mfs() -> None:
    x, y = _make_dataset(40)
    est = HDFISProdClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    assert all(isinstance(mf, DimensionDependentGaussianMF) for mfs in est.model_.input_mfs.values() for mf in mfs)


def test_hdfis_prod_regressor_estimator_uses_dimension_dependent_mfs() -> None:
    x = np.random.RandomState(0).normal(size=(40, 3)).astype(np.float32)
    y = (x[:, 0] * 0.5 + x[:, 1] * -0.2).astype(np.float32)
    est = HDFISProdRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    assert all(isinstance(mf, DimensionDependentGaussianMF) for mfs in est.model_.input_mfs.values() for mf in mfs)


def test_hdfis_min_classifier_estimator_builds_hdfis_min_model() -> None:
    x, y = _make_dataset(40)
    est = HDFISMinClassifier(
        n_mfs=2,
        mf_init="kmeans",
        epochs=2,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    assert isinstance(est.model_, HDFISMinClassifierModel)
    assert all(not p.requires_grad for p in est.model_.membership_layer.parameters())


def test_hdfis_min_regressor_estimator_builds_hdfis_min_model() -> None:
    x = np.random.RandomState(0).normal(size=(40, 3)).astype(np.float32)
    y = (x[:, 0] * 0.5 + x[:, 1] * -0.2).astype(np.float32)
    est = HDFISMinRegressor(
        n_mfs=2,
        mf_init="kmeans",
        epochs=2,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
    )

    est.fit(x, y)
    assert isinstance(est.model_, HDFISMinRegressorModel)
    assert all(not p.requires_grad for p in est.model_.membership_layer.parameters())


def test_hdfis_prod_classifier_paper_strict_uses_paper_protocol_defaults() -> None:
    x, _ = _make_dataset(40)
    est = HDFISProdClassifier(paper_strict=True, epochs=1, random_state=7)

    input_mfs, feature_names, effective_rule_base = est._build_input_mfs(x)

    assert est.mf_init == "grid"
    assert est.n_mfs == 3
    assert est.batch_size == 64
    assert effective_rule_base == "coco"
    assert feature_names == ["x1", "x2", "x3"]
    assert all(len(input_mfs[name]) == 3 for name in feature_names)
    assert all(isinstance(mf, DimensionDependentGaussianMF) for mfs in input_mfs.values() for mf in mfs)
    assert all(getattr(mf, "paper_strict_equation", False) for mfs in input_mfs.values() for mf in mfs)


def test_hdfis_min_classifier_paper_strict_uses_paper_protocol_defaults() -> None:
    x, _ = _make_dataset(40)
    est = HDFISMinClassifier(paper_strict=True, epochs=1, random_state=7)

    input_mfs, _, effective_rule_base = est._build_input_mfs(x)

    assert est.mf_init == "grid"
    assert est.n_mfs == 3
    assert est.batch_size == 64
    assert effective_rule_base == "coco"
    assert all(isinstance(mf, GaussianMF) for mfs in input_mfs.values() for mf in mfs)


def test_hdfis_paper_strict_rejects_conflicting_hyperparameters() -> None:
    with pytest.raises(ValueError, match="paper_strict requires n_mfs=3"):
        HDFISProdClassifier(paper_strict=True, n_mfs=4)
    with pytest.raises(ValueError, match="paper_strict requires mf_init='grid'"):
        HDFISProdClassifier(paper_strict=True, mf_init="kmeans")
    with pytest.raises(ValueError, match="paper_strict requires rule_base='coco'"):
        HDFISMinRegressor(paper_strict=True, rule_base="cartesian")
    with pytest.raises(ValueError, match="paper_strict requires batch_size=64"):
        HDFISMinClassifier(paper_strict=True, batch_size=32)
