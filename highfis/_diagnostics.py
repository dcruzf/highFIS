"""Numerical diagnostics of the rule firing of a fitted model."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

#: Rows of the training data used for the check made at the end of ``fit``.
_FIT_CHECK_ROWS = 2000


class DegenerateFiringWarning(UserWarning):
    """The rule weights of a fitted model carry no information about the input."""


def firing_diagnostics(
    weights: np.ndarray,
    *,
    dominance: float = 0.99,
    never: float = 1e-6,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Summarize normalized rule weights of shape ``(samples, rules)``.

    See :meth:`highfis.estimators.BaseTSKEstimator.firing_diagnostics` for the meaning of
    the returned entries.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] == 0 or w.shape[1] == 0:
        raise ValueError(f"expected rule weights of shape (samples, rules), got {w.shape}")
    n_samples, n_rules = w.shape
    finite = np.isfinite(w).all(axis=1)
    safe = np.where(np.isfinite(w), w, 0.0)

    spread = safe.max(axis=1) - safe.min(axis=1)
    uniform = finite & (spread <= tolerance) if n_rules > 1 else np.zeros(n_samples, dtype=bool)
    dominated = finite & (safe.max(axis=1) >= dominance)

    total = safe.sum(axis=1, keepdims=True)
    p = np.divide(safe, total, out=np.zeros_like(safe), where=total > 0)
    entropy = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)
    effective = np.exp(entropy)

    mean_firing = safe.mean(axis=0)
    never_firing = np.flatnonzero(safe.max(axis=0) < never)

    return {
        "n_samples": int(n_samples),
        "n_rules": int(n_rules),
        "uniform_fraction": float(uniform.mean()),
        "dominated_fraction": float(dominated.mean()),
        "non_finite_fraction": float(1.0 - finite.mean()),
        "effective_rules": float(effective.mean()),
        "effective_rules_min": float(effective.min()),
        "mean_firing": mean_firing,
        "never_firing_rules": never_firing,
    }


def warn_if_degenerate(report: dict[str, Any]) -> None:
    """Warn when every rule of a fitted model has the same weight on most samples.

    The firing strength of every rule has underflowed, and the model is a single linear
    model whatever the number of rules. One rule taking almost all the weight is not
    reported: some families concentrate the weight by construction on high-dimensional
    data, and ``firing_diagnostics`` shows it when asked.
    """
    n_rules = int(report["n_rules"])
    if n_rules < 2:
        return
    if report["uniform_fraction"] > 0.5:
        warnings.warn(
            f"Every rule has the same weight on {100 * report['uniform_fraction']:.0f}% of the training "
            f"samples, so the {n_rules} rules act as a single linear model. The firing strength of every "
            "rule underflows: with many features, use a family built for high-dimensional data (for "
            "example HTSK) or reduce the number of features; with few features, the fuzzy sets are too "
            "narrow for the data, so increase sigma_scale. See firing_diagnostics().",
            DegenerateFiringWarning,
            stacklevel=3,
        )
