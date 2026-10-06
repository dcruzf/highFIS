"""Diagnostic plots for fitted estimators.

Every function takes a fitted estimator and draws on matplotlib axes. The same plots are
available as methods: ``estimator.plot(kind=...)`` and ``estimator.plot_<kind>(...)``.

Available kinds:
    - ``"memberships"`` — the learned fuzzy sets, one panel per feature.
    - ``"history"`` — loss or a metric per epoch, with the validation curve and the
      best epoch when they were recorded.
    - ``"rule_activation"`` — mean normalized firing strength of each rule, split by
      class when labels are given.
    - ``"diagnostics"`` — confusion matrix and decision margin for classifiers;
      predicted against observed and residuals for regressors.

Conventions:
    - matplotlib is an optional dependency (``pip install highfis[plot]``) and is only
      imported when a plot is requested.
    - Single-panel plots accept ``ax`` and return the ``Axes``. Multi-panel plots accept a
      sequence of axes in ``ax`` and return the ``Figure``.
    - No function calls ``plt.show()`` or changes ``rcParams``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from sklearn.base import is_classifier
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.utils.validation import check_is_fitted

PLOT_KINDS: tuple[str, ...] = ("memberships", "history", "rule_activation", "diagnostics")

_PHASE_LABELS = {"dg": "DG", "fs": "feature selection", "re": "rule extraction", "finetune": "fine-tune"}
_HEATMAP_MAX_ANNOTATED_CELLS = 80
_MAX_LEGEND_ENTRIES = 6


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install it with `pip install highfis[plot]`.") from exc
    return plt


def _axes(ax: Any, n_panels: int, ncols: int, panel_size: tuple[float, float]) -> tuple[Any, list[Any]]:
    """Return ``(figure, axes)`` for *n_panels*, creating a figure unless *ax* is given."""
    if ax is not None:
        axes = list(np.atleast_1d(np.asarray(ax, dtype=object)).ravel())
        if len(axes) != n_panels:
            raise ValueError(f"this plot needs {n_panels} axes; got {len(axes)}")
        return axes[0].figure, axes
    nrows = -(-n_panels // ncols)
    fig, grid = _pyplot().subplots(
        nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False, layout="constrained"
    )
    axes = list(grid.ravel())
    for unused in axes[n_panels:]:
        unused.set_visible(False)
    return fig, axes[:n_panels]


def _model_columns(estimator: Any) -> np.ndarray:
    """Original column index of each model input, in model order.

    Families that prune features keep a subset of the columns seen in ``fit``.
    """
    columns = np.arange(int(estimator.n_features_in_))[None, :]
    return np.asarray(estimator._select_model_features(columns)[0], dtype=int)


def _feature_labels(estimator: Any) -> list[str]:
    """Display name of each model input: the name seen in ``fit`` when there is one."""
    names = getattr(estimator, "feature_names_in_", None)
    model_names = [str(name) for name in estimator.model_.input_names]
    if names is None:
        return model_names
    return [str(names[column]) for column in _model_columns(estimator)]


def _select_features(estimator: Any, features: Sequence[int | str] | None, max_features: int) -> list[int]:
    """Positions, among the model inputs, of the features to draw."""
    labels = _feature_labels(estimator)
    columns = _model_columns(estimator).tolist()
    if features is None:
        if len(labels) <= max_features:
            return list(range(len(labels)))
        importance = estimator.feature_importance()
        if importance is None:
            return list(range(max_features))
        return sorted(np.argsort(-np.asarray(importance))[:max_features].tolist())

    positions = []
    for feature in features:
        if isinstance(feature, str):
            if feature not in labels:
                raise ValueError(f"unknown feature {feature!r}; the fitted model uses {labels}")
            positions.append(labels.index(feature))
        elif int(feature) in columns:
            positions.append(columns.index(int(feature)))
        else:
            raise ValueError(f"feature index {feature} is not used by the fitted model; it uses columns {columns}")
    return positions


def _mf_extent(params: dict[str, Any]) -> tuple[float, float] | None:
    """Interval that shows a fuzzy set, read from its parameters."""
    centre = params.get("mean", params.get("center"))
    if centre is not None and "sigma" in params:
        return centre - 3.0 * params["sigma"], centre + 3.0 * params["sigma"]
    if centre is not None and {"left", "right"} <= params.keys():
        return params["left"], params["right"]
    if centre is not None and "a" in params:
        return centre - 3.0 * params["a"], centre + 3.0 * params["a"]
    if {"a", "d"} <= params.keys():
        return params["a"], params["d"]
    return None


def _feature_range(mf_params: list[dict[str, Any]], column: np.ndarray | None) -> tuple[float, float]:
    if column is not None:
        lo, hi = float(np.min(column)), float(np.max(column))
        pad = 0.05 * (hi - lo) or 0.5
        return lo - pad, hi + pad
    extents = [extent for extent in map(_mf_extent, mf_params) if extent is not None]
    if not extents:
        return 0.0, 1.0
    return float(min(lo for lo, _ in extents)), float(max(hi for _, hi in extents))


def plot_memberships(
    estimator: Any,
    features: Sequence[int | str] | None = None,
    *,
    X: Any = None,
    max_features: int = 6,
    n_points: int = 200,
    ax: Any = None,
) -> Any:
    """Plot the learned fuzzy sets, one panel per feature.

    Args:
        estimator: A fitted estimator.
        features: Features to show, as names or as column indices of the data seen in
            ``fit``. ``None`` shows every feature, or the ``max_features`` most important
            ones when the model has more.
        X: Optional data. When given, each panel spans the range of its column; otherwise
            the range is taken from the parameters of the fuzzy sets.
        max_features: Number of features shown when ``features`` is ``None``.
        n_points: Number of points used to draw each curve.
        ax: Axes to draw on, one per feature. ``None`` creates a new figure.

    Returns:
        The ``Figure`` holding the panels.
    """
    check_is_fitted(estimator, "model_")
    positions = _select_features(estimator, features, max_features)
    labels = _feature_labels(estimator)
    model_names = [str(name) for name in estimator.model_.input_names]
    mf_params = estimator.get_mf_params()
    input_mfs = estimator.model_.membership_layer.input_mfs
    data = None if X is None else estimator._select_model_features(np.asarray(X))

    fig, axes = _axes(ax, len(positions), ncols=min(3, len(positions)), panel_size=(3.2, 2.4))
    for panel, position in zip(axes, positions, strict=True):
        name = model_names[position]
        lo, hi = _feature_range(mf_params[name], None if data is None else data[:, position])
        # The centres are added to the grid so that the peak of a narrow set is not missed.
        centres = [p.get("mean", p.get("center")) for p in mf_params[name]]
        centres = [c for c in centres if c is not None and lo <= c <= hi]
        points = np.unique(np.concatenate([np.linspace(lo, hi, n_points), centres]))
        grid = torch.as_tensor(points, dtype=estimator._model_dtype(), device=str(estimator.device))
        sets = [(j, mf) for j, mf in enumerate(input_mfs[name]) if type(mf).__name__ != "ConstantMF"]
        with torch.no_grad():
            for j, mf in sets:
                panel.plot(grid.cpu().numpy(), mf(grid).cpu().numpy(), label=f"MF{j}")
        panel.set_title(labels[position])
        panel.set_ylim(0.0, 1.05)
        panel.set_xlabel("Feature value")
        panel.set_ylabel("Membership degree")
        if 0 < len(sets) <= _MAX_LEGEND_ENTRIES:
            panel.legend()
    return fig


def _history_phases(history: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Split ``history_`` into training phases; single-phase families give one unnamed phase."""
    if "train_loss" in history:
        return [("", history)]
    return [
        (_PHASE_LABELS[key], history[key])
        for key in _PHASE_LABELS
        if isinstance(history.get(key), dict) and "train_loss" in history[key]
    ]


def _available_metrics(phases: list[tuple[str, dict[str, Any]]]) -> list[str]:
    reserved = {"loss", "ur_loss", "total_loss"}
    keys = {key[len("train_") :] for _, phase in phases for key in phase if key.startswith("train_")}
    return sorted(keys - reserved)


def plot_history(estimator: Any, metric: str | None = None, *, ax: Any = None) -> Any:
    """Plot the training loss, or a metric, per epoch.

    The validation curve is drawn when a validation set was passed to ``fit``, and the
    best epoch is marked when early stopping recorded one. Families that train in several
    phases (DG-TSK, DG-ALETSK, FSRE-ADATSK) show the phases one after the other.

    Args:
        estimator: A fitted estimator.
        metric: Name of a metric recorded during ``fit`` (for example ``"accuracy"``).
            ``None`` plots the loss.
        ax: Axes to draw on. ``None`` creates a new figure.

    Returns:
        The ``Axes`` with the curves.
    """
    check_is_fitted(estimator, "model_")
    phases = _history_phases(estimator.history_)
    name = "loss" if metric is None else metric
    if metric is not None and metric not in _available_metrics(phases):
        raise ValueError(f"metric {metric!r} was not recorded during fit; available: {_available_metrics(phases)}")

    _, (panel,) = _axes(ax, 1, ncols=1, panel_size=(4.8, 3.2))
    offset = 0
    for index, (label, phase) in enumerate(phases):
        n_epochs = len(phase["train_loss"])
        for prefix, colour in (("train", "C0"), ("val", "C1")):
            series = phase.get(f"{prefix}_{name}")
            if not series:
                continue
            # Metrics may be evaluated every few epochs; spread them over the phase.
            epochs = offset + (np.arange(len(series)) + 1) * n_epochs / len(series)
            legend = {"train": "training", "val": "validation"}[prefix] if index == 0 else None
            panel.plot(epochs, series, color=colour, label=legend)
        best = phase.get("best_epoch")
        if best is not None:
            panel.axvline(offset + best + 1, color="0.4", linestyle="--", label="best epoch" if index == 0 else None)
        if label:
            panel.axvline(offset + 0.5, color="0.8", linewidth=0.8)
            panel.text(offset + 0.5, 1.01, f" {label}", transform=panel.get_xaxis_transform(), va="bottom")
        offset += n_epochs
    panel.set_xlabel("Epoch")
    panel.set_ylabel(name.replace("_", " ").capitalize())
    panel.legend()
    return panel


def _annotated_heatmap(panel: Any, values: np.ndarray, cmap: str, fmt: Callable[[float], str]) -> None:
    panel.imshow(values, cmap=cmap, aspect="auto", vmin=0.0)
    if values.size > _HEATMAP_MAX_ANNOTATED_CELLS:
        return
    threshold = float(values.max()) / 2.0
    for (row, col), value in np.ndenumerate(values):
        panel.text(col, row, fmt(value), ha="center", va="center", color="white" if value >= threshold else "black")


def plot_rule_activation(estimator: Any, X: Any, y: Any = None, *, ax: Any = None) -> Any:
    """Plot the mean normalized firing strength of each rule.

    Args:
        estimator: A fitted estimator.
        X: Data on which the rules are evaluated.
        y: Optional class labels, for classifiers only. When given, the mean activation is
            computed separately for the samples of each class.
        ax: Axes to draw on. ``None`` creates a new figure.

    Returns:
        The ``Axes`` with the heatmap (labels given) or the bar chart (no labels).
    """
    check_is_fitted(estimator, "model_")
    activation = estimator.rule_activation(X)
    n_rules = activation.shape[1]
    rule_labels = [f"Rule {r}" for r in range(n_rules)]
    _, (panel,) = _axes(ax, 1, ncols=1, panel_size=(4.8, max(2.4, 0.3 * n_rules + 1.2)))

    if y is None:
        panel.barh(np.arange(n_rules), activation.mean(axis=0))
        panel.set_xlabel("Mean rule activation")
        panel.invert_yaxis()
    else:
        if not is_classifier(estimator):
            raise ValueError("y groups the activations by class, so it only applies to classifiers")
        y = np.asarray(y)
        classes = estimator.classes_
        by_class = np.stack([activation[y == c].mean(axis=0) for c in classes], axis=1)
        _annotated_heatmap(panel, by_class, "Greens", lambda v: f"{v:.2f}")
        panel.set_xticks(np.arange(len(classes)), [str(c) for c in classes])
        panel.set_xlabel("True class")
        panel.set_title("Mean rule activation")
    if n_rules <= 30:
        panel.set_yticks(np.arange(n_rules), rule_labels)
    else:
        panel.set_ylabel("Rule")
    return panel


def _classification_diagnostics(estimator: Any, X: Any, y: np.ndarray, axes: list[Any]) -> None:
    classes = estimator.classes_
    labels = [str(c) for c in classes]
    ticks = np.arange(len(classes))
    proba = estimator.predict_proba(X)
    predicted = classes[np.argmax(proba, axis=1)]

    matrix = confusion_matrix(y, predicted, labels=classes)
    _annotated_heatmap(axes[0], matrix, "Oranges", lambda v: str(int(v)))
    axes[0].set_xticks(ticks, labels)
    axes[0].set_yticks(ticks, labels)
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_title("Confusion matrix")

    # Margin of a sample: probability of its true class minus the best other class.
    # It is negative exactly when the sample is misclassified.
    true_index = np.searchsorted(classes, y)
    rows = np.arange(len(y))
    true_proba = proba[rows, true_index]
    others = proba.copy()
    others[rows, true_index] = -np.inf
    margin = true_proba - others.max(axis=1)
    rng = np.random.default_rng(0)
    for tick in ticks:
        values = margin[true_index == tick]
        axes[1].scatter(tick + rng.uniform(-0.18, 0.18, size=len(values)), values, alpha=0.7, edgecolor="black", lw=0.3)
    axes[1].axhline(0.0, color="0.4", linestyle="--", label="decision boundary")
    axes[1].set_xticks(ticks, labels)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("True class")
    axes[1].set_ylabel("P(true class) $-$ P(best other class)")
    axes[1].set_title("Decision margin")
    axes[1].legend()


def _regression_diagnostics(estimator: Any, X: Any, y: np.ndarray, axes: list[Any]) -> None:
    predicted = estimator.predict(X)
    lo, hi = float(min(y.min(), predicted.min())), float(max(y.max(), predicted.max()))
    pad = 0.05 * (hi - lo) or 0.5

    axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.4", linestyle="--", label="ideal")
    axes[0].scatter(y, predicted, alpha=0.7, edgecolor="black", lw=0.3, label=f"$R^2$ = {r2_score(y, predicted):.3f}")
    axes[0].set_xlabel("Observed")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted against observed")
    axes[0].legend()

    axes[1].axhline(0.0, color="0.4", linestyle="--")
    axes[1].scatter(y, predicted - y, alpha=0.7, edgecolor="black", lw=0.3)
    axes[1].set_xlabel("Observed")
    axes[1].set_ylabel("Residual (predicted $-$ observed)")
    axes[1].set_title("Residuals")


def plot_diagnostics(estimator: Any, X: Any, y: Any, *, ax: Any = None) -> Any:
    """Plot two views of the predictions on ``(X, y)``.

    For a classifier: the confusion matrix, and the decision margin of every sample
    grouped by its true class. The margin is the probability given to the true class minus
    the highest probability given to another class, so it is negative for the samples that
    are misclassified.

    For a regressor: predicted against observed values, and the residuals against the
    observed values.

    Args:
        estimator: A fitted estimator.
        X: Data to predict.
        y: True labels or targets for ``X``.
        ax: Two axes to draw on. ``None`` creates a new figure.

    Returns:
        The ``Figure`` holding the two panels.
    """
    check_is_fitted(estimator, "model_")
    fig, axes = _axes(ax, 2, ncols=2, panel_size=(3.6, 3.2))
    draw = _classification_diagnostics if is_classifier(estimator) else _regression_diagnostics
    draw(estimator, X, np.asarray(y), axes)
    return fig


_PLOTTERS: dict[str, Callable[..., Any]] = {
    "memberships": plot_memberships,
    "history": plot_history,
    "rule_activation": plot_rule_activation,
    "diagnostics": plot_diagnostics,
}


def plot(estimator: Any, kind: str = "memberships", **kwargs: Any) -> Any:
    """Draw the plot named *kind* for a fitted estimator.

    Args:
        estimator: A fitted estimator.
        kind: One of ``"memberships"`` (default), ``"history"``, ``"rule_activation"``
            or ``"diagnostics"``.
        **kwargs: Arguments of the corresponding ``plot_<kind>`` function.

    Returns:
        The ``Axes`` or ``Figure`` returned by that function.
    """
    if kind not in _PLOTTERS:
        raise ValueError(f"kind must be one of {list(PLOT_KINDS)}; got {kind!r}")
    return _PLOTTERS[kind](estimator, **kwargs)


__all__: list[str] = [
    "PLOT_KINDS",
    "plot",
    "plot_diagnostics",
    "plot_history",
    "plot_memberships",
    "plot_rule_activation",
]
