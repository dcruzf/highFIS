"""Human-readable descriptions of a fitted estimator: names, rules, summary and explanations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_LEVELS: dict[int, tuple[str, ...]] = {
    2: ("low", "high"),
    3: ("low", "medium", "high"),
    5: ("very low", "low", "medium", "high", "very high"),
}


class Report(str):
    """Text that shows itself, and carries the numbers it was written from.

    A report is a string: ``print`` it, search it, write it to a file. At an interactive
    prompt or as the last line of a notebook cell it is displayed as the text itself, with
    its line breaks, so no ``print`` is needed there. ``data`` holds what the text
    describes, for the code that needs the values and not their description.
    """

    data: dict[str, Any]

    def __new__(cls, text: str, data: dict[str, Any] | None = None) -> Report:
        """Create the report from its text and from the values it describes."""
        report = super().__new__(cls, text)
        report.data = dict(data or {})
        return report

    def __repr__(self) -> str:
        """Return the text itself, which is what an interactive session displays."""
        return str.__str__(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the values that the text describes."""
        return dict(self.data)


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
) -> Report:
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
    data = {
        "features": names,
        "classes": None if classes is None else list(classes),
        "fuzzy_sets": {names[i]: sets[model_names[i]] for i in range(len(names))},
        "rules": [{names[i]: sets[model_names[i]][rule[model_names[i]]] for i in range(len(names))} for rule in table],
        "bias": bias,
        "weights": weights,
    }
    return Report("\n".join(lines), data)


_PHASES = {"dg": "gates", "fs": "feature selection", "re": "rule extraction", "finetune": "fine-tuning"}


def _phases(history: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Training phases recorded in ``history_``, in the order they ran."""
    found = [(label, history[key]) for key, label in _PHASES.items() if isinstance(history.get(key), dict)]
    return found or [("", history)]


def _phase_line(phase: dict[str, Any], digits: int) -> str:
    losses = phase.get("train_loss") or []
    config = phase.get("config") or {}
    parts = [f"{len(losses)} epochs"]
    if "learning_rate" in config:
        parts.append(f"learning rate {config['learning_rate']:g}")
    if losses:
        parts.append(f"training loss {losses[0]:.{digits}f} -> {losses[-1]:.{digits}f}")
    for key, values in phase.items():
        reserved = key in ("train_loss", "train_ur_loss", "train_total_loss")
        if key.startswith("train_") and not reserved and values:
            parts.append(f"training {key[len('train_') :].replace('_', ' ')} {values[-1]:.{digits}f}")
    if phase.get("val_loss"):
        parts.append(f"validation loss {phase['val_loss'][-1]:.{digits}f}")
    if phase.get("best_epoch") is not None:
        parts.append(f"best epoch {int(phase['best_epoch']) + 1}")
    return ", ".join(parts)


def summary(estimator: Any, digits: int = 4) -> Report:
    """Describe a fitted estimator: structure, training configuration and training outcome.

    See :meth:`highfis.estimators._base._BaseTSKEstimator.summary`.
    """
    model = estimator.model_
    classes = getattr(estimator, "classes_", None)
    n_seen, n_used = int(estimator.n_features_in_), int(model.n_inputs)
    sets = sorted({len(mfs) for mfs in model.membership_layer.input_mfs.values()})
    kinds = sorted({type(mf).__name__ for mfs in model.membership_layer.input_mfs.values() for mf in mfs})
    rule_layer = type(model.rule_layer).__name__
    if rule_layer == "RuleLayer":
        t_norm = getattr(model.rule_layer, "_resolved_t_norm", None)
        rule_layer = f"RuleLayer, T-norm {getattr(t_norm, '__name__', type(t_norm).__name__)}"

    # The rule base asked for, when one was named: a gated family prunes it during ``fit``.
    asked = getattr(estimator, "rule_base", None)
    rule_base = asked if isinstance(asked, str) else getattr(estimator, "rule_base_", "unknown")

    def count(module: Any) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    antecedent, consequent = count(model.membership_layer), count(model.consequent_layer)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rows = [
        ("task", "regression" if classes is None else f"classification, {len(classes)} classes"),
        ("features", f"{n_seen} seen in fit, {n_used} used by the model"),
        ("rules", f"{int(model.n_rules)} (rule base: {rule_base})"),
        ("fuzzy sets", f"{' or '.join(str(n) for n in sets)} per feature, {', '.join(kinds)}"),
        ("firing strength", rule_layer),
        ("defuzzifier", type(model.defuzzifier).__name__),
        ("consequents", type(model.consequent_layer).__name__),
        ("parameters", f"{total} trainable ({antecedent} antecedent, {consequent} consequent)"),
    ]
    batch = getattr(estimator, "batch_size_", None)
    training = [("batch size", "full batch" if batch is None else str(batch))]
    for label, phase in _phases(estimator.history_):
        training.append((label or "epochs", _phase_line(phase, digits)))
    solved = estimator.history_.get("lse") if isinstance(estimator.history_, dict) else None
    if solved:
        training.append(
            (
                "least squares",
                f"consequents solved at the end, ridge {solved['ridge']:g}, "
                f"training error {solved['train_mse']:.{digits}f}",
            )
        )

    lines = [type(estimator).__name__]
    lines += [f"  {name:18s} {value}" for name, value in rows]
    lines.append("Training")
    lines += [f"  {name:18s} {value}" for name, value in training]
    data = {
        "estimator": type(estimator).__name__,
        "task": "regression" if classes is None else "classification",
        "classes": None if classes is None else list(classes),
        "n_features_in": n_seen,
        "n_features_used": n_used,
        "n_rules": int(model.n_rules),
        "rule_base": rule_base,
        "n_fuzzy_sets": sets,
        "membership_functions": kinds,
        "n_parameters": {"total": total, "antecedent": antecedent, "consequent": consequent},
        "batch_size": batch,
        "phases": [
            {
                "name": label or "training",
                "epochs": len(phase.get("train_loss") or []),
                "first_loss": (phase.get("train_loss") or [None])[0],
                "last_loss": (phase.get("train_loss") or [None])[-1],
                "best_epoch": phase.get("best_epoch"),
            }
            for label, phase in _phases(estimator.history_)
        ],
        "least_squares": dict(solved) if solved else None,
    }
    return Report("\n".join(lines), data)


def explain(
    estimator: Any,
    weights: np.ndarray,
    contributions: np.ndarray,
    prediction: Any,
    top_rules: int | None = 3,
    top_features: int | None = 3,
    digits: int = 2,
    labels: Sequence[str] | None = None,
) -> Report:
    """Describe how the rules produce the prediction of one sample.

    See :meth:`highfis.estimators._base._BaseTSKEstimator.explain`.
    """
    model = estimator.model_
    names = feature_labels(estimator)
    model_names = [str(name) for name in model.input_names]
    mf_params = estimator.get_mf_params()
    sets = {name: set_labels(mf_params[name], labels) for name in model_names}
    importance = estimator.feature_importance()
    ranked = list(range(len(names))) if importance is None else np.argsort(-np.asarray(importance)).tolist()
    shown = ranked if top_features is None else ranked[:top_features]
    classes = getattr(estimator, "classes_", None)
    table = model.get_rule_table()

    def scores(values: np.ndarray) -> str:
        if classes is None:
            return f"{float(values[0]):+.{digits}f}"
        return ", ".join(f"{label}: {float(value):+.{digits}f}" for label, value in zip(classes, values, strict=True))

    order = np.argsort(-weights)
    kept = order if top_rules is None else order[:top_rules]
    shown_prediction = f"{float(prediction):.{digits}f}" if classes is None else str(prediction)
    lines = [f"Prediction: {shown_prediction}"]
    lines.append(f"Output, the sum of the contributions of the {len(order)} rules: {scores(contributions.sum(axis=0))}")
    for r in kept:
        lines.append(f"Rule {int(r)}, weight {float(weights[r]):.{digits}f}: contributes {scores(contributions[r])}")
        lines.append(f"    IF {_antecedent(table[int(r)], model_names, names, sets, shown)}")
    if len(kept) < len(order):
        rest = np.setdiff1d(order, kept)
        lines.append(
            f"{len(rest)} other rule{'s' if len(rest) > 1 else ''}, weight {float(weights[rest].sum()):.{digits}f} "
            f"in all: contribute{'' if len(rest) > 1 else 's'} {scores(contributions[rest].sum(axis=0))}"
        )
    data = {
        "prediction": prediction,
        "classes": None if classes is None else list(classes),
        "output": contributions.sum(axis=0),
        "weights": weights,
        "contributions": contributions,
    }
    return Report("\n".join(lines), data)
