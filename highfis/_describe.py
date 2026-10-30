"""Human-readable descriptions of a fitted estimator: names, rules, summary and explanations."""

from __future__ import annotations

import sys
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


def interactive_session() -> bool:
    """Whether the code runs at a prompt or in a notebook, where a result displays itself."""
    if hasattr(sys, "ps1") or bool(sys.flags.interactive):
        return True
    ipython = sys.modules.get("IPython")
    return ipython is not None and ipython.get_ipython() is not None


def deliver(report: Report, show: bool | None) -> Report:
    """Print *report* when asked to, and return it.

    With ``show=None`` the report is printed in a script and left alone in an interactive
    session or a notebook, where it displays itself when it is the value of the last line:
    the same call then shows the text however the code is run.
    """
    if show is None:
        show = not interactive_session()
    if show:
        sys.stdout.write(f"{report}\n")
    return report


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


def _pack(prefix: str, parts: Sequence[str], separator: str, width: int, indent: str | None = None) -> list[str]:
    """Lay *parts* out after *prefix*, breaking lines between parts and never inside one.

    The lines that continue the first are indented to the width of the prefix, or with
    *indent* when it is given. A separator that is a punctuation mark stays at the end of
    the line it closes; a word such as ``AND`` opens the next line.
    """
    indent = " " * len(prefix) if indent is None else indent
    lines, current = [], prefix
    for index, part in enumerate(parts):
        piece = part if index == 0 else separator + part
        if not current.strip() or len(current) + len(piece) <= width:
            current += piece
        elif separator[:1].strip():
            lines.append(current + separator.rstrip())
            current = indent + part
        else:
            lines.append(current.rstrip())
            current = indent + piece.lstrip()
    lines.append(current.rstrip())
    return lines


def _signed(value: float, digits: int) -> str:
    """Format with a sign, writing a value that rounds to zero as ``+0.00`` and never ``-0.00``."""
    text = f"{value:+.{digits}f}"
    return f"{0.0:+.{digits}f}" if float(text) == 0.0 else text


def _linear_parts(
    bias: float | None, weights: np.ndarray | None, names: list[str], top: int | None, digits: int
) -> list[str]:
    """Terms of ``b + w1 * x1 ...``, keeping the *top* coefficients with the largest magnitude."""
    parts = [] if bias is None else [f"{bias:.{digits}f}"]
    if weights is not None:
        order = np.argsort(-np.abs(weights))
        kept = order if top is None else order[:top]
        for index in kept:
            sign = "-" if weights[index] < 0 else "+"
            parts.append(f"{sign} {abs(weights[index]):.{digits}f} * {names[index]}")
        if len(kept) < len(order):
            # Kept with the last term, so that a line never holds the ellipsis alone.
            parts[-1] += " + ..."
    if not parts:
        return ["(no consequent parameters)"]
    parts[0] = parts[0].removeprefix("+ ")
    return parts


def _conditions(
    rule: dict[str, Any], model_names: list[str], names: list[str], sets: dict[str, list[str | None]]
) -> list[str]:
    """Conditions of a rule, as the parts of ``a is low AND b is high``.

    When more than half of the features share a fuzzy set, the rule is written as the
    features that differ followed by ``the other N features are low``: that is what tells a
    rule from the next in a rule base on many features.
    """
    active = [
        (names[position], label)
        for position, name in enumerate(model_names)
        if (label := sets[name][rule[name]]) is not None
    ]
    if not active:
        return ["(always)"]
    labels = [label for _, label in active]
    common = max(sorted(set(labels)), key=labels.count)
    shared = labels.count(common)
    if len(active) <= 3 or 2 * shared <= len(active):
        return [f"{name} is {label}" for name, label in active]
    if shared == len(active):
        return [f"all {shared} features are {common}"]
    parts = [f"{name} is {label}" for name, label in active if label != common]
    rest = "the other feature is" if shared == 1 else f"the other {shared} features are"
    return [*parts, f"{rest} {common}"]


def rules_as_text(
    estimator: Any,
    top_features: int | None = 5,
    digits: int = 2,
    labels: Sequence[str] | None = None,
    max_rules: int | None = None,
    width: int = 80,
) -> Report:
    """Describe the fitted rule base as ``IF ... THEN ...`` sentences.

    See :meth:`highfis.estimators._base._BaseTSKEstimator.rules_as_text`.
    """
    model = estimator.model_
    names = feature_labels(estimator)
    model_names = [str(name) for name in model.input_names]
    mf_params = estimator.get_mf_params()
    sets = {name: set_labels(mf_params[name], labels) for name in model_names}

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
        lines += _pack(f"{head}IF ", _conditions(rule, model_names, names, sets), " AND ", width)
        if classes is None:
            b = None if bias is None else float(bias[r])
            w = None if weights is None else weights[r]
            lines += _pack(f"{indent}THEN y = ", _linear_parts(b, w, names, top_features, digits), " ", width)
            continue
        longest = max(len(str(label)) for label in classes)
        for c, label in enumerate(classes):
            b = None if bias is None else float(bias[r, c])
            w = None if weights is None else weights[r, c]
            prefix = f"{indent}{'THEN' if c == 0 else '    '} {str(label) + ':':{longest + 1}s} "
            lines += _pack(prefix, _linear_parts(b, w, names, top_features, digits), " ", width)
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


def _training_metrics(phase: dict[str, Any]) -> dict[str, float]:
    """Last value of each training metric recorded in a phase, by the name of the metric."""
    reserved = ("train_loss", "train_ur_loss", "train_total_loss")
    return {
        key[len("train_") :].replace("_", " "): values[-1]
        for key, values in phase.items()
        if key.startswith("train_") and key not in reserved and values
    }


def _training_table(phases: list[tuple[str, dict[str, Any]]], digits: int) -> list[str]:
    """One row per training phase, one column per quantity that at least one phase recorded."""
    columns: list[tuple[str, list[str]]] = []

    def column(title: str, cells: list[str]) -> None:
        if any(cells):
            columns.append((title, [cell or "-" for cell in cells]))

    histories = [phase for _, phase in phases]
    losses = [phase.get("train_loss") or [] for phase in histories]
    column("epochs", [str(len(values)) for values in losses])
    rates = [(phase.get("config") or {}).get("learning_rate") for phase in histories]
    column("rate", ["" if rate is None else f"{rate:g}" for rate in rates])
    column("loss first -> last", [f"{v[0]:.{digits}f} -> {v[-1]:.{digits}f}" if v else "" for v in losses])
    metrics = [_training_metrics(phase) for phase in histories]
    for name in sorted({name for recorded in metrics for name in recorded}):
        column(name, [f"{recorded[name]:.{digits}f}" if name in recorded else "" for recorded in metrics])
    validation = [phase.get("val_loss") or [] for phase in histories]
    column("val. loss", [f"{values[-1]:.{digits}f}" if values else "" for values in validation])
    best = [phase.get("best_epoch") for phase in histories]
    column("best epoch", ["" if epoch is None else str(int(epoch) + 1) for epoch in best])

    names = [label or "training" for label, _ in phases]
    first = max(len("phase"), *(len(name) for name in names))
    widths = [max(len(title), *(len(cell) for cell in cells)) for title, cells in columns]
    header = "".join(f"  {title:>{w}s}" for (title, _), w in zip(columns, widths, strict=True))
    rows = [f"  {'phase':{first}s}{header}"]
    for index, name in enumerate(names):
        cells = "".join(f"  {cells[index]:>{w}s}" for (_, cells), w in zip(columns, widths, strict=True))
        rows.append(f"  {name:{first}s}{cells}")
    return rows


def summary(estimator: Any, digits: int = 4, width: int = 80) -> Report:
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
    # FSRE-ADATSK replaces it during ``fit``, and the model records the one it ended with.
    asked = getattr(estimator, "rule_base", None)
    rule_base = asked if isinstance(asked, str) else getattr(estimator, "rule_base_", "unknown")
    rule_base = getattr(model, "rule_base_name", rule_base)

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
    phases = _phases(estimator.history_)
    solved = estimator.history_.get("lse") if isinstance(estimator.history_, dict) else None

    lines = [type(estimator).__name__]
    for name, value in rows:
        lines += _pack(f"  {name:18s} ", value.split(" "), " ", width)
    lines.append(f"Training ({'full batch' if batch is None else f'batch size {batch}'})")
    lines += _training_table(phases, digits)
    if solved:
        text = (
            f"consequents solved at the end, ridge {solved['ridge']:g}, training error {solved['train_mse']:.{digits}f}"
        )
        lines += _pack("  least squares: ", text.split(" "), " ", width)
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
            for label, phase in phases
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
    digits: int = 2,
    labels: Sequence[str] | None = None,
    width: int = 80,
) -> Report:
    """Describe how the rules produce the prediction of one sample.

    See :meth:`highfis.estimators._base._BaseTSKEstimator.explain`.
    """
    model = estimator.model_
    names = feature_labels(estimator)
    model_names = [str(name) for name in model.input_names]
    mf_params = estimator.get_mf_params()
    sets = {name: set_labels(mf_params[name], labels) for name in model_names}
    classes = getattr(estimator, "classes_", None)
    table = model.get_rule_table()

    def scores(values: np.ndarray) -> list[str]:
        if classes is None:
            return [_signed(float(values[0]), digits)]
        pairs = zip(classes, values, strict=True)
        return [f"{label}: {_signed(float(value), digits)}" for label, value in pairs]

    order = np.argsort(-weights)
    # A rule whose weight is written as zero says nothing about this sample: it goes with the rest.
    visible = [int(r) for r in order if float(weights[r]) >= 0.5 * 10.0**-digits]
    kept = [int(r) for r in order] if top_rules is None else visible[:top_rules]
    shown_prediction = f"{float(prediction):.{digits}f}" if classes is None else str(prediction)
    lines = [f"Prediction: {shown_prediction}"]
    total = f"Output, the sum of the contributions of the {len(order)} rules: "
    lines += _pack(total, scores(contributions.sum(axis=0)), ", ", width, indent="    ")
    for r in kept:
        head = f"Rule {r}, weight {float(weights[r]):.{digits}f}: contributes "
        lines += _pack(head, scores(contributions[r]), ", ", width, indent="    ")
        lines += _pack("    IF ", _conditions(table[r], model_names, names, sets), " AND ", width)
    if len(kept) < len(order):
        rest = np.setdiff1d(order, kept)
        several = len(rest) > 1
        head = (
            f"{len(rest)} other rule{'s' if several else ''}, weight {float(weights[rest].sum()):.{digits}f} "
            f"in all: contribute{'' if several else 's'} "
        )
        lines += _pack(head, scores(contributions[rest].sum(axis=0)), ", ", width, indent="    ")
    data = {
        "prediction": prediction,
        "classes": None if classes is None else list(classes),
        "output": contributions.sum(axis=0),
        "weights": weights,
        "contributions": contributions,
    }
    return Report("\n".join(lines), data)
