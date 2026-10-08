"""Rule-firing diagnostics and the warning for degenerate rule weights."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from highfis import DegenerateFiringWarning, HTSKClassifier, HTSKRegressor, TSKClassifier
from highfis._diagnostics import firing_diagnostics, warn_if_degenerate


def _wide_data(n_features: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((60, n_features)).astype(np.float32)
    return x, (x[:, 0] > 0.5).astype(int)


def test_report_of_known_weights() -> None:
    weights = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.75, 0.0]])
    report = firing_diagnostics(weights)
    assert report["n_samples"] == 4
    assert report["n_rules"] == 3
    assert report["uniform_fraction"] == 0.0
    assert report["dominated_fraction"] == 0.25
    assert report["non_finite_fraction"] == 0.0
    assert report["effective_rules_min"] == pytest.approx(1.0)
    assert 1.0 < report["effective_rules"] < 2.0
    np.testing.assert_allclose(report["mean_firing"], [0.5625, 0.4375, 0.0])
    assert report["never_firing_rules"].tolist() == [2]


def test_uniform_weights_are_reported() -> None:
    report = firing_diagnostics(np.full((5, 4), 0.25))
    assert report["uniform_fraction"] == 1.0
    assert report["effective_rules"] == pytest.approx(4.0)
    assert report["never_firing_rules"].size == 0


def test_non_finite_weights_are_counted() -> None:
    weights = np.array([[0.5, 0.5], [np.nan, 0.5]])
    assert firing_diagnostics(weights)["non_finite_fraction"] == 0.5


def test_a_single_rule_is_never_uniform() -> None:
    report = firing_diagnostics(np.ones((3, 1)))
    assert report["uniform_fraction"] == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_degenerate(report)


def test_rejects_weights_that_are_not_a_matrix() -> None:
    with pytest.raises(ValueError, match="samples, rules"):
        firing_diagnostics(np.ones(3))


def test_one_rule_taking_all_the_weight_warns() -> None:
    weights = np.tile([0.999, 0.001], (10, 1))
    with pytest.warns(DegenerateFiringWarning, match="Rule 0 takes more than 99%"):
        warn_if_degenerate(firing_diagnostics(weights))


def test_the_product_underflows_in_high_dimension() -> None:
    x, y = _wide_data()
    with pytest.warns(DegenerateFiringWarning, match="same weight"):
        model = TSKClassifier(n_mfs=3, epochs=2, random_state=0).fit(x, y)
    report = model.firing_diagnostics(x)
    assert report["uniform_fraction"] == 1.0
    assert report["effective_rules"] == pytest.approx(3.0)


def test_a_family_for_high_dimension_does_not_warn() -> None:
    x, y = _wide_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DegenerateFiringWarning)
        model = HTSKClassifier(n_mfs=3, epochs=2, random_state=0).fit(x, y)
    report = model.firing_diagnostics(x)
    assert report["uniform_fraction"] == 0.0
    assert report["n_rules"] == 3


def test_regressors_have_the_method() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((40, 5)).astype(np.float32)
    model = HTSKRegressor(n_mfs=2, epochs=2, random_state=0).fit(x, x[:, 0])
    report = model.firing_diagnostics(x)
    assert report["n_samples"] == 40
    assert report["mean_firing"].shape == (2,)
    assert report["mean_firing"].sum() == pytest.approx(1.0, abs=1e-5)
