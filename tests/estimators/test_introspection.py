"""Introspection methods must work on every estimator, including those that prune features."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

import highfis

ESTIMATORS: list[str] = sorted(n for n in highfis.__all__ if n.endswith(("Classifier", "Regressor")))


def _fit(name: str) -> tuple[Any, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((80, 6))
    y = (x[:, 0] > 0.5).astype(int) if name.endswith("Classifier") else 2.0 * x[:, 0]
    cls = getattr(highfis, name)
    epochs = {p: 3 for p in inspect.signature(cls.__init__).parameters if p.endswith("epochs")}
    return cls(random_state=0, **epochs).fit(x, y), x


@pytest.mark.parametrize("name", ESTIMATORS)
def test_rule_activation_matches_the_fitted_model(name: str) -> None:
    """``rule_activation`` used to fail on FSRE-ADATSK and on the DG regressors.

    Those families drop features during ``fit``; the inputs have to be sliced to the
    surviving ones before they reach the model, as ``predict`` already did.
    """
    est, x = _fit(name)

    activation = est.rule_activation(x[:4])

    assert activation.shape == (4, est.model_.n_rules)
    assert np.isfinite(activation).all()


def test_the_sweep_includes_models_that_pruned_features() -> None:
    """Guard the test above: without a pruned model it would not exercise the slicing."""
    pruned = [name for name in ESTIMATORS if _fit(name)[0].model_.n_inputs < 6]

    assert {"FSREADATSKClassifier", "FSREADATSKRegressor", "DGTSKRegressor"} <= set(pruned)
