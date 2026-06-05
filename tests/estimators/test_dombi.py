from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from highfis import DombiTSKClassifier, DombiTSKRegressor
from highfis.memberships import GaussianMF, GaussianPiMF


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


def test_dombi_tsk_classifier_estimator_fit_predict() -> None:
    x, y = _make_dataset(60)
    est = DombiTSKClassifier(
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


def test_dombi_tsk_regressor_estimator_fit_predict_score() -> None:
    x, y = _make_regression_dataset(80)
    est = DombiTSKRegressor(
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


def test_dombi_tsk_classifier_paper_strict_uses_paper_protocol_defaults() -> None:
    est = DombiTSKClassifier(paper_strict=True)
    assert est.n_mfs == 3
    assert est.mf_init == "grid"
    assert est.sigma_scale == 1.0
    assert est.rule_base == "coco"
    assert est.lambda_ == 1.0
    assert abs(est.lower_bound - 1.0 / math.e) < 1e-6
    assert est.zero_consequent_init is True


def test_dombi_tsk_classifier_paper_strict_rejects_conflicting_hyperparameters() -> None:
    with pytest.raises(ValueError, match=r"paper_strict requires n_mfs=3"):
        DombiTSKClassifier(paper_strict=True, n_mfs=5)
    with pytest.raises(ValueError, match=r"paper_strict requires mf_init='grid'"):
        DombiTSKClassifier(paper_strict=True, mf_init="kmeans")
    with pytest.raises(ValueError, match=r"paper_strict requires sigma_scale=1.0"):
        DombiTSKClassifier(paper_strict=True, sigma_scale=2.0)
    with pytest.raises(ValueError, match=r"paper_strict requires rule_base='coco'"):
        DombiTSKClassifier(paper_strict=True, rule_base="cartesian")
    with pytest.raises(ValueError, match=r"paper_strict requires lambda_=1.0"):
        DombiTSKClassifier(paper_strict=True, lambda_=2.0)
    with pytest.raises(ValueError, match=r"paper_strict requires lower_bound=1/e"):
        DombiTSKClassifier(paper_strict=True, lower_bound=0.5)
    with pytest.raises(ValueError, match=r"paper_strict requires zero_consequent_init=True"):
        DombiTSKClassifier(paper_strict=True, zero_consequent_init=False)


def test_dombi_tsk_classifier_paper_strict_builds_correct_mfs_and_zeros() -> None:
    x = np.random.randn(45, 2)
    y = np.random.randint(0, 2, (45,))
    est = DombiTSKClassifier(paper_strict=True, epochs=1)

    # 1. Verify model zero-initialization before fit
    input_mfs_for_build = {
        "x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
        "x2": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    }
    model = est._build_model(input_mfs_for_build, n_classes=2, rule_base="coco")
    weight = getattr(model.consequent_layer, "weight", None)
    bias = getattr(model.consequent_layer, "bias", None)
    assert weight is not None
    assert torch.all(weight == 0.0)
    assert bias is not None
    assert torch.all(bias == 0.0)

    # 2. Fit the estimator
    est.fit(x, y)

    # Verify Composite Gaussian MFs wrapping (which uses GaussianPiMF)
    for mfs in est.model_.input_mfs.values():
        for mf in mfs:
            assert isinstance(mf, GaussianPiMF)
            assert abs(mf.eps - 1.0 / math.e) < 1e-6
