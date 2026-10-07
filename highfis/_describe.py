"""Human-readable descriptions of a fitted estimator: feature names and rules as text."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_LEVELS: dict[int, tuple[str, ...]] = {
    2: ("low", "high"),
    3: ("low", "medium", "high"),
    5: ("very low", "low", "medium", "high", "very high"),
}


def model_columns(estimator: Any) -> np.ndarray:
    """Original column index of each model input, in model order.

    Families that prune features keep a subset of the columns seen in ``fit``.
    """
    columns = np.arange(int(estimator.n_features_in_))[None, :]
    return np.asarray(estimator._select_model_features(columns)[0], dtype=int)


def feature_labels(estimator: Any) -> list[str]:
    """Display name of each model input: the name seen in ``fit`` when there is one."""
    names = getattr(estimator, "feature_names_in_", None)
    if names is None:
        return [str(name) for name in estimator.model_.input_names]
    return [str(names[column]) for column in model_columns(estimator)]


def _set_centre(params: dict[str, Any]) -> float | None:
    """Location of a fuzzy set, or ``None`` for a set that does not depend on the input."""
    centre = params.get("mean", params.get("center"))
    if centre is not None:
        return float(centre)
    if {"b", "c"} <= params.keys():  # trapezoid: middle of the plateau
        return (float(params["b"]) + float(params["c"])) / 2.0
    return None


def set_labels(mf_params: Sequence[dict[str, Any]], labels: Sequence[str] | None = None) -> list[str | None]:
    """Name each fuzzy set of one feature from the order of the set centres.

    Two, three and five sets are named ``low`` ... ``high``; other counts are named
    ``level i of n``. *labels* replaces the names when its length matches the number of
    sets. A set without a location (a "don't care" set) gets ``None``.
    """
    centres = [_set_centre(params) for params in mf_params]
    located = [i for i, centre in enumerate(centres) if centre is not None]
    order = sorted(located, key=lambda i: centres[i])  # type: ignore[arg-type, return-value]
    n = len(order)
    if labels is not None and len(labels) == n:
        names: Sequence[str] = labels
    else:
        names = _LEVELS.get(n, tuple(f"level {rank + 1} of {n}" for rank in range(n)))
    result: list[str | None] = [None] * len(centres)
    for rank, index in enumerate(order):
        result[index] = names[rank]
    return result


def _linear_terms(
    bias: float | None, weights: np.ndarray | None, names: list[str], top: int | None, digits: int
) -> str:
    """Format ``b + w1 * x1 ...`` keeping the *top* coefficients with the largest magnitude."""
    parts = [] if bias is None else [f"{bias:.{digits}f}"]
    if weights is not None:
        order = np.argsort(-np.abs(weights))
        kept = order if top is None else order[:top]
        for index in kept:
            sign = "-" if weights[index] < 0 else "+"
            parts.append(f"{sign} {abs(weights[index]):.{digits}f} * {names[index]}")
        if len(kept) < len(order):
            parts.append("+ ...")
    if not parts:
        return "(no consequent parameters)"
    return " ".join(parts).removeprefix("+ ")


def _antecedent(
    rule: dict[str, Any],
    model_names: list[str],
    names: list[str],
    sets: dict[str, list[str | None]],
    shown: list[int],
) -> str:
    conditions = [
        (position, sets[model_names[position]][rule[model_names[position]]]) for position in range(len(model_names))
    ]
    active = [(position, label) for position, label in conditions if label is not None]
    kept = [f"{names[position]} is {label}" for position, label in active if position in shown]
    hidden = len(active) - len(kept)
    if hidden:
        kept.append(f"... ({hidden} more)")
    return " AND ".join(kept) if kept else "(always)"


def rules_as_text(
    estimator: Any,
    top_features: int | None = 5,
    digits: int = 2,
    labels: Sequence[str] | None = None,
    max_rules: int | None = None,
) -> str:
    """Describe the fitted rule base as ``IF ... THEN ...`` sentences.

    See :meth:`highfis.estimators._base._BaseTSKEstimator.rules_as_text`.
    """
    model = estimator.model_
    names = feature_labels(estimator)
    model_names = [str(name) for name in model.input_names]
    mf_params = estimator.get_mf_params()
    sets = {name: set_labels(mf_params[name], labels) for name in model_names}

    importance = estimator.feature_importance()
    ranked = list(range(len(names))) if importance is None else np.argsort(-np.asarray(importance)).tolist()
    shown = ranked if top_features is None else ranked[:top_features]

    weights = estimator.get_consequent_weights()
    bias = estimator.get_consequent_bias()
    classes = getattr(estimator, "classes_", None)
    table = model.get_rule_table()
    rules = table if max_rules is None else table[:max_rules]

    lines = []
    for rule in rules:
        r = rule["rule_id"]
        head = f"Rule {r}: "
        indent = " " * len(head)
        lines.append(f"{head}IF {_antecedent(rule, model_names, names, sets, shown)}")
        if classes is None:
            b = None if bias is None else float(bias[r])
            w = None if weights is None else weights[r]
            lines.append(f"{indent}THEN y = {_linear_terms(b, w, names, top_features, digits)}")
            continue
        for c, label in enumerate(classes):
            b = None if bias is None else float(bias[r, c])
            w = None if weights is None else weights[r, c]
            prefix = "THEN " if c == 0 else "     "
            lines.append(f"{indent}{prefix}{label}: {_linear_terms(b, w, names, top_features, digits)}")
    if len(rules) < len(table):
        lines.append(f"... ({len(table) - len(rules)} more rules)")
    return "\n".join(lines)
