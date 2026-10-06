"""Run scikit-learn's estimator checks against every exported estimator.

``parametrize_with_checks`` exercises the full estimator contract -- input validation,
fitted attributes, pickling, ``clone``, pipeline and meta-estimator behaviour -- which the
``get_params`` round trips in ``test_sklearn_params`` only cover in part.

The checks are run on short trainings to keep the sweep fast, with two exceptions that are
handled in :func:`test_sklearn_estimator_checks`.

The sweep is still about 1500 tests, so it is deselected by default. Run it with::

    hatch test -m sklearn_checks
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import parametrize_with_checks

import highfis

pytestmark = pytest.mark.sklearn_checks

ESTIMATORS: list[str] = sorted(n for n in highfis.__all__ if n.endswith(("Classifier", "Regressor")))

# Most checks only need a fitted model, not a good one.
FAST_EPOCHS = 5

# These two assert a minimum accuracy / R^2, so the model has to actually converge -- unless
# the estimator declares the ``poor_score`` tag, which makes scikit-learn skip the threshold.
TRAINING_QUALITY_CHECKS = {"check_classifiers_train", "check_regressors_train"}
TRAINING_QUALITY_EPOCHS = 20
# Estimators that need longer to clear the threshold with a margin; ``None`` keeps the
# default schedule.
TRAINING_QUALITY_OVERRIDES: dict[str, int | None] = {
    "ADATSKRegressor": None,
    "FSREADATSKRegressor": None,
    "MHTSKRegressor": 40,
}

# Compares predictions at ``rtol=1e-7``, the resolution of float32 itself: a row's result
# moves by one ulp with its position in the batch. The invariance is checked in float64.
DOUBLE_PRECISION_CHECKS = {"check_methods_sample_order_invariance"}


def _epoch_params(cls: type) -> list[str]:
    return [p for p in inspect.signature(cls.__init__).parameters if p.endswith("epochs")]


def _make(name: str, epochs: int | None) -> BaseEstimator:
    """Build *name* with every ``*epochs`` argument set to *epochs* (``None`` keeps defaults)."""
    cls = getattr(highfis, name)
    overrides = {} if epochs is None else dict.fromkeys(_epoch_params(cls), epochs)
    return cls(random_state=0, **overrides)


def _check_name(check: Callable[..., Any]) -> str:
    return getattr(check, "func", check).__name__


def _has_poor_score(estimator: Any) -> bool:
    tags = estimator.__sklearn_tags__()
    task_tags = tags.classifier_tags or tags.regressor_tags
    return bool(task_tags.poor_score)


@pytest.fixture
def float64_default() -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def test_estimators_are_discovered() -> None:
    """Guard the sweep itself: an empty list would make the checks below vacuous."""
    assert len(ESTIMATORS) >= 28


@parametrize_with_checks([_make(name, FAST_EPOCHS) for name in ESTIMATORS])  # type: ignore[misc]
def test_sklearn_estimator_checks(
    estimator: BaseEstimator, check: Callable[..., Any], request: pytest.FixtureRequest
) -> None:
    """Every estimator passes every applicable scikit-learn check."""
    name = _check_name(check)
    if name in TRAINING_QUALITY_CHECKS and not _has_poor_score(estimator):
        cls_name = type(estimator).__name__
        estimator = _make(cls_name, TRAINING_QUALITY_OVERRIDES.get(cls_name, TRAINING_QUALITY_EPOCHS))
    if name in DOUBLE_PRECISION_CHECKS:
        request.getfixturevalue("float64_default")
    check(estimator)
