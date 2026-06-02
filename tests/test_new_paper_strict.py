"""Tests for newly implemented paper_strict modes (TSK, LogTSK, DombiTSK)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from highfis import (
    DombiTSKClassifier,
    LogTSKClassifier,
    LogTSKRegressor,
    TSKClassifier,
    TSKRegressor,
)
from highfis.estimators._htsk import _HTSKPaperStrictTrainer
from highfis.memberships import GaussianMF, GaussianPiMF


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


def test_tsk_and_logtsk_paper_strict_trainers() -> None:
    x = np.random.randn(45, 2)
    y = np.random.randint(0, 2, (45,))

    # Test TSK (use default 200 epochs as locked by paper_strict)
    est_tsk = TSKClassifier(paper_strict=True)
    est_tsk.fit(x, y)
    assert isinstance(est_tsk._get_trainer(), _HTSKPaperStrictTrainer)
    assert est_tsk.history_["stopped_epoch"] is not None

    # Test LogTSK (use default 200 epochs as locked by paper_strict)
    est_log = LogTSKClassifier(paper_strict=True)
    est_log.fit(x, y)
    assert isinstance(est_log._get_trainer(), _HTSKPaperStrictTrainer)
    assert est_log.history_["stopped_epoch"] is not None
