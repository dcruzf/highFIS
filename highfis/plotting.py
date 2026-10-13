"""Diagnostic plots for fitted estimators.

Every function takes a fitted estimator and draws on matplotlib axes. The same plots are
available as methods: ``estimator.plot(kind=...)`` and ``estimator.plot_<kind>(...)``.

Available kinds:
    - ``"memberships"`` — the learned fuzzy sets, one panel per feature.
    - ``"history"`` — loss or a metric per epoch, with the validation curve and the
      best epoch when they were recorded.
    - ``"rules"`` — the rule base as a table: the fuzzy set that each rule uses on each
      feature.
    - ``"rule_activation"`` — mean normalized firing strength of each rule, split by
      class when labels are given.
    - ``"diagnostics"`` — confusion matrix and decision margin for classifiers;
      predicted against observed and residuals for regressors.
    - ``"feature_importance"`` — the features that weigh most in the consequents.
    - ``"gates"`` — the gates of the features and rules that a gated family kept, with
      the thresholds they passed.

Conventions:
    - matplotlib is an optional dependency (``pip install highfis[plot]``) and is only
      imported when a plot is requested.
    - Single-panel plots accept ``ax`` and return the ``Axes``. Multi-panel plots accept a
      sequence of axes in ``ax`` and return the ``Figure``.
    - In an interactive session or a notebook a plot is displayed and nothing is returned,
      as with a plot command; in a script, or with ``show=False``, the ``Axes`` or the
      ``Figure`` is returned and nothing is displayed.
    - The colours come from one theme, changed with :func:`set_plot_theme`, and each plot
      takes ``colors`` or ``cmap`` for a single figure. No function changes ``rcParams``.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from sklearn.base import is_classifier
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.utils.validation import check_is_fitted

from ._describe import feature_labels as _feature_labels
from ._describe import model_columns as _model_columns
from ._describe import set_labels as _set_labels

PLOT_KINDS: tuple[str, ...] = (
    "memberships",
    "history",
    "rules",
    "rule_activation",
    "diagnostics",
    "feature_importance",
    "gates",
)

#: Colours of Okabe and Ito (2008), which stay distinct for readers with a colour vision
#: deficiency. Yellow comes last: it is the weakest on a white page.
_OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442")


@dataclass(frozen=True)
class PlotTheme:
    """Colours shared by every plot of the package.

    Attributes:
        colors: Colours of the things that are told apart by name: the fuzzy sets of a
            feature, in the membership plot and in the table of the rule base, the training
            and validation curves, the classes.
        sequential: Colormap of the tables whose cells are amounts, from little to much:
            the confusion matrix and the rule activation.
        reference: Colour of the lines that mark a reference: thresholds, the decision
            boundary, the best epoch.
    """

    colors: tuple[str, ...] = _OKABE_ITO
    sequential: str = "Blues"
    reference: str = "0.4"


_theme = PlotTheme()


def get_plot_theme() -> PlotTheme:
    """Return the colours that the plots use."""
    return _theme


def set_plot_theme(
    *,
    colors: Sequence[str] | None = None,
    sequential: str | None = None,
    reference: str | None = None,
) -> PlotTheme:
    """Change the colours of every plot drawn from now on.

    Only the arguments given are changed. Colours are anything matplotlib accepts as a
    colour, and colormaps are names of matplotlib colormaps.

    Args:
        colors: Colours of the things told apart by name, in the order they are used.
        sequential: Colormap of the confusion matrix and of the rule activation.
        reference: Colour of the reference lines.

    Returns:
        The theme now in use.
    """
    global _theme
    changes: dict[str, Any] = {"sequential": sequential, "reference": reference}
    if colors is not None:
        if len(colors) == 0:
            raise ValueError("colors must have at least one colour")
        changes["colors"] = tuple(colors)
    _theme = replace(_theme, **{key: value for key, value in changes.items() if value is not None})
    return _theme


def reset_plot_theme() -> PlotTheme:
    """Go back to the colours of the package."""
    global _theme
    _theme = PlotTheme()
    return _theme


def _colours(colors: Sequence[str] | None) -> tuple[str, ...]:
    """Colours of one plot: those of the call when given, those of the theme otherwise."""
    if colors is None:
        return _theme.colors
    if len(colors) == 0:
        raise ValueError("colors must have at least one colour")
    return tuple(colors)


def _text_colour(image: Any, value: float) -> str:
    """Black or white, whichever reads better on the cell that *image* draws for *value*."""
    red, green, blue, _ = image.cmap(image.norm(value))
    return "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"


_PHASE_LABELS = {"dg": "DG", "fs": "feature selection", "re": "rule extraction", "finetune": "fine-tune"}
_HEATMAP_MAX_ANNOTATED_CELLS = 80
_RULE_TABLE_MAX_ANNOTATED_CELLS = 400
_MAX_LEGEND_ENTRIES = 6


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install it with `pip install highfis[plot]`.") from exc
    return plt


def _interactive_session() -> bool:
    """Whether the code runs at a prompt or in a notebook, where a figure is meant to be seen."""
    if hasattr(sys, "ps1") or bool(sys.flags.interactive):
        return True
    ipython = sys.modules.get("IPython")
    return ipython is not None and ipython.get_ipython() is not None


def _shown(function: Callable[..., Any]) -> Callable[..., Any]:
    """Give a plot function the ``show`` argument.

    ``show=True`` displays the figure and returns nothing, as a plot command does;
    ``show=False`` returns the ``Axes`` or the ``Figure`` without displaying anything. The
    default, ``None``, displays in an interactive session or a notebook, unless the caller
    passed its own axes, and returns the artist in a script, which is therefore never
    stopped by a window.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, show: bool | None = None, **kwargs: Any) -> Any:
        artist = function(*args, **kwargs)
        if show is None:
            show = kwargs.get("ax") is None and _interactive_session()
        if not show:
            return artist
        _pyplot().show()
        return None

    return wrapper


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


def _set_position(params: dict[str, Any]) -> float:
    """Location of a fuzzy set on the axis of its feature, to order the sets of a feature."""
    centre = params.get("mean", params.get("center"))
    if centre is None and {"b", "c"} <= params.keys():
        centre = (params["b"] + params["c"]) / 2.0
    return float(centre) if centre is not None else 0.0


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


@_shown
def plot_memberships(
    estimator: Any,
    features: Sequence[int | str] | None = None,
    *,
    X: Any = None,
    max_features: int = 6,
    n_points: int = 200,
    set_names: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
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
        set_names: Names of the fuzzy sets of a feature, from the lowest to the highest,
            used when their number matches. By default two, three and five sets are named
            ``low`` ... ``high``, as in :meth:`rules_as_text`.
        colors: Colours of the fuzzy sets, from the lowest to the highest. ``None`` uses the
            theme (see :func:`set_plot_theme`).
        ax: Axes to draw on, one per feature. ``None`` creates a new figure.
        show: Display the figure and return nothing (``True``), or return it without
            displaying (``False``). ``None`` displays in an interactive session or a
            notebook, and returns it in a script or when ``ax`` is given.

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
    legends: list[list[str]] = []
    palette = _colours(colors)
    for panel, position in zip(axes, positions, strict=True):
        name = model_names[position]
        lo, hi = _feature_range(mf_params[name], None if data is None else data[:, position])
        # The centres are added to the grid so that the peak of a narrow set is not missed.
        centres = [p.get("mean", p.get("center")) for p in mf_params[name]]
        centres = [c for c in centres if c is not None and lo <= c <= hi]
        points = np.unique(np.concatenate([np.linspace(lo, hi, n_points), centres]))
        grid = torch.as_tensor(points, dtype=estimator._model_dtype(), device=str(estimator.device))
        names = _set_labels(mf_params[name], set_names)
        located = [j for j in range(len(names)) if names[j] is not None]
        # The colour follows the position of a set among those of its feature, from the
        # lowest to the highest, so that a name has the same colour on every panel.
        rank = {j: i for i, j in enumerate(sorted(located, key=lambda j: _set_position(mf_params[name][j])))}
        with torch.no_grad():
            for j in located:
                curve = input_mfs[name][j](grid).cpu().numpy()
                panel.plot(grid.cpu().numpy(), curve, label=names[j], color=palette[rank[j] % len(palette)])
        legends.append([str(names[j]) for j in sorted(located, key=rank.__getitem__)])
        panel.set_title(labels[position])
        panel.set_ylim(0.0, 1.05)
        panel.set_xlabel("Feature value")
        panel.set_ylabel("Membership degree")
    # The same sets on every panel: one legend beside the panels, where it hides no curve.
    shared = ax is None and len(set(map(tuple, legends))) == 1 and 0 < len(legends[0]) <= _MAX_LEGEND_ENTRIES
    for panel, entries in zip(axes[:1] if shared else axes, legends, strict=False):
        if not 0 < len(entries) <= _MAX_LEGEND_ENTRIES:
            continue
        by_label = dict(zip(*reversed(panel.get_legend_handles_labels()), strict=True))
        handles = [by_label[entry] for entry in entries]
        if shared:
            fig.legend(handles, entries, loc="outside right upper", title="Fuzzy set")
        else:
            panel.legend(handles, entries, fontsize="small")
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


def _draw_phase(panel: Any, phase: dict[str, Any], name: str, palette: Sequence[str]) -> None:
    """Draw the training and validation curves of one training phase, and its best epoch."""
    n_epochs = len(phase["train_loss"])
    for prefix, colour in (("train", palette[0]), ("val", palette[1 % len(palette)])):
        series = phase.get(f"{prefix}_{name}")
        if not series:
            continue
        # Metrics may be evaluated every few epochs; spread them over the phase.
        epochs = (np.arange(len(series)) + 1) * n_epochs / len(series)
        panel.plot(epochs, series, color=colour, label={"train": "training", "val": "validation"}[prefix])
    best = phase.get("best_epoch")
    if best is not None:
        panel.axvline(best + 1, color=_theme.reference, linestyle="--", label="best epoch")
    panel.set_xlabel("Epoch")


@_shown
def plot_history(
    estimator: Any, metric: str | None = None, *, colors: Sequence[str] | None = None, ax: Any = None
) -> Any:
    """Plot the training loss, or a metric, per epoch.

    The validation curve is drawn when a validation set was passed to ``fit``, and the
    best epoch is marked when early stopping recorded one. Families that train in several
    phases (DG-TSK, DG-ALETSK, FSRE-ADATSK) get one panel per phase, side by side and on
    the same vertical scale.

    Args:
        estimator: A fitted estimator.
        metric: Name of a metric recorded during ``fit`` (for example ``"accuracy"``).
            ``None`` plots the loss.
        colors: Colours of the training and of the validation curve. ``None`` uses the
            theme (see :func:`set_plot_theme`).
        ax: Axes to draw on, one per training phase. ``None`` creates a new figure.
        show: Display the figure and return nothing (``True``), or return it without
            displaying (``False``). ``None`` displays in an interactive session or a
            notebook, and returns it in a script or when ``ax`` is given.

    Returns:
        The ``Axes`` with the curves, or the ``Figure`` holding one panel per phase for a
        family that trains in several phases.
    """
    check_is_fitted(estimator, "model_")
    phases = _history_phases(estimator.history_)
    name = "loss" if metric is None else metric
    if metric is not None and metric not in _available_metrics(phases):
        raise ValueError(f"metric {metric!r} was not recorded during fit; available: {_available_metrics(phases)}")

    several = len(phases) > 1
    fig, panels = _axes(ax, len(phases), ncols=len(phases), panel_size=(3.4, 3.0) if several else (4.8, 3.2))
    for index, (panel, (label, phase)) in enumerate(zip(panels, phases, strict=True)):
        _draw_phase(panel, phase, name, _colours(colors))
        if label:
            panel.set_title(label[0].upper() + label[1:])
        if index == 0:
            panel.set_ylabel(name.replace("_", " ").capitalize())
            if panel.get_legend_handles_labels()[0]:
                panel.legend()
    if several:
        # One vertical scale, so that the level reached by a phase can be compared with the next.
        limits = [panel.get_ylim() for panel in panels]
        for panel in panels:
            panel.set_ylim(min(lo for lo, _ in limits), max(hi for _, hi in limits))
        return fig
    return panels[0]


def _annotated_heatmap(panel: Any, values: np.ndarray, cmap: str | None, fmt: Callable[[float], str]) -> None:
    image = panel.imshow(values, cmap=cmap or _theme.sequential, aspect="auto", vmin=0.0)
    if values.size > _HEATMAP_MAX_ANNOTATED_CELLS:
        return
    for (row, col), value in np.ndenumerate(values):
        panel.text(col, row, fmt(value), ha="center", va="center", color=_text_colour(image, float(value)))


@_shown
def plot_rule_activation(
    estimator: Any,
    X: Any,
    y: Any = None,
    *,
    cmap: str | None = None,
    colors: Sequence[str] | None = None,
    ax: Any = None,
) -> Any:
    """Plot the mean normalized firing strength of each rule.

    Args:
        estimator: A fitted estimator.
        X: Data on which the rules are evaluated.
        y: Optional class labels, for classifiers only. When given, the mean activation is
            computed separately for the samples of each class.
        cmap: Colormap of the table drawn when ``y`` is given. ``None`` uses the sequential
            colormap of the theme (see :func:`set_plot_theme`).
        colors: Colour of the bars drawn when ``y`` is not given, as a sequence whose first
            colour is used. ``None`` uses the theme.
        ax: Axes to draw on. ``None`` creates a new figure.
        show: Display the figure and return nothing (``True``), or return it without
            displaying (``False``). ``None`` displays in an interactive session or a
            notebook, and returns it in a script or when ``ax`` is given.

    Returns:
        The ``Axes`` with the heatmap (labels given) or the bar chart (no labels).
    """
    check_is_fitted(estimator, "model_")
    activation = estimator.rule_activation(X)
    n_rules = activation.shape[1]
    rule_labels = [f"Rule {r}" for r in range(n_rules)]
    _, (panel,) = _axes(ax, 1, ncols=1, panel_size=(4.8, max(2.4, 0.3 * n_rules + 1.2)))

    if y is None:
        panel.barh(np.arange(n_rules), activation.mean(axis=0), color=_colours(colors)[0])
        panel.set_xlabel("Mean rule activation")
        panel.invert_yaxis()
    else:
        if not is_classifier(estimator):
            raise ValueError("y groups the activations by class, so it only applies to classifiers")
        y = np.asarray(y)
        classes = estimator.classes_
        by_class = np.stack([activation[y == c].mean(axis=0) for c in classes], axis=1)
        _annotated_heatmap(panel, by_class, cmap, lambda v: f"{v:.2f}")
        panel.set_xticks(np.arange(len(classes)), [str(c) for c in classes])
        panel.set_xlabel("True class")
        panel.set_title("Mean rule activation")
    if n_rules <= 30:
        panel.set_yticks(np.arange(n_rules), rule_labels)
    else:
        panel.set_ylabel("Rule")
    return panel


def _classification_diagnostics(
    estimator: Any, X: Any, y: np.ndarray, axes: list[Any], cmap: str | None, palette: Sequence[str]
) -> None:
    classes = estimator.classes_
    labels = [str(c) for c in classes]
    ticks = np.arange(len(classes))
    proba = estimator.predict_proba(X)
    predicted = classes[np.argmax(proba, axis=1)]

    matrix = confusion_matrix(y, predicted, labels=classes)
    _annotated_heatmap(axes[0], matrix, cmap, lambda v: str(int(v)))
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
        jitter = tick + rng.uniform(-0.18, 0.18, size=len(values))
        axes[1].scatter(jitter, values, color=palette[tick % len(palette)], alpha=0.7, edgecolor="black", lw=0.3)
    axes[1].axhline(0.0, color=_theme.reference, linestyle="--", label="decision boundary")
    axes[1].set_xticks(ticks, labels)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("True class")
    axes[1].set_ylabel("P(true class) $-$ P(best other class)")
    axes[1].set_title("Decision margin")
    axes[1].legend()


def _regression_diagnostics(
    estimator: Any, X: Any, y: np.ndarray, axes: list[Any], cmap: str | None, palette: Sequence[str]
) -> None:
    predicted = estimator.predict(X)
    lo, hi = float(min(y.min(), predicted.min())), float(max(y.max(), predicted.max()))
    pad = 0.05 * (hi - lo) or 0.5

    points = {"color": palette[0], "alpha": 0.7, "edgecolor": "black", "lw": 0.3}
    axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=_theme.reference, linestyle="--", label="ideal")
    axes[0].scatter(y, predicted, label=f"$R^2$ = {r2_score(y, predicted):.3f}", **points)
    axes[0].set_xlabel("Observed")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted against observed")
    axes[0].legend()

    axes[1].axhline(0.0, color=_theme.reference, linestyle="--")
    axes[1].scatter(y, predicted - y, **points)
    axes[1].set_xlabel("Observed")
    axes[1].set_ylabel("Residual (predicted $-$ observed)")
    axes[1].set_title("Residuals")


@_shown
def plot_diagnostics(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    cmap: str | None = None,
    colors: Sequence[str] | None = None,
    ax: Any = None,
) -> Any:
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
        cmap: Colormap of the confusion matrix. ``None`` uses the sequential colormap of
            the theme (see :func:`set_plot_theme`).
        colors: Colours of the classes in the decision margin, or of the points of a
            regressor. ``None`` uses the theme.
        ax: Two axes to draw on. ``None`` creates a new figure.
        show: Display the figure and return nothing (``True``), or return it without
            displaying (``False``). ``None`` displays in an interactive session or a
            notebook, and returns it in a script or when ``ax`` is given.

    Returns:
        The ``Figure`` holding the two panels.
    """
    check_is_fitted(estimator, "model_")
    fig, axes = _axes(ax, 2, ncols=2, panel_size=(3.6, 3.2))
    draw = _classification_diagnostics if is_classifier(estimator) else _regression_diagnostics
    draw(estimator, X, np.asarray(y), axes, cmap, _colours(colors))
    return fig


@_shown
def plot_feature_importance(
    estimator: Any, *, top: int | None = 20, colors: Sequence[str] | None = None, ax: Any = None
) -> Any:
    """Plot the features that weigh most in the consequents of the fitted model.

    The importance is the one of ``estimator.feature_importance()``: the mean magnitude of
    the consequent coefficients of each feature, normalized to sum to one. Features are
    named as in ``fit`` when the data had names, and by their column otherwise.

    Args:
        estimator: A fitted estimator.
        top: Number of features to draw, from the most important. ``None`` draws all.
        colors: Colour of the bars, as a sequence whose first colour is used. ``None`` uses
            the theme (see :func:`set_plot_theme`).
        ax: Axes to draw on. ``None`` creates a new figure.
        show: Display the figure and return nothing (``True``), or return it without
            displaying (``False``). ``None`` displays in an interactive session or a
            notebook, and returns it in a script or when ``ax`` is given.

    Returns:
        The ``Axes`` with the bar chart.
    """
    check_is_fitted(estimator, "model_")
    importance = estimator.feature_importance()
    if importance is None:
        raise ValueError("this model has no consequent coefficients to compute a feature importance from")
    importance = np.asarray(importance, dtype=float)
    labels = _feature_labels(estimator)
    order = np.argsort(-importance)
    if top is not None:
        order = order[: int(top)]

    _, (panel,) = _axes(ax, 1, ncols=1, panel_size=(4.8, max(2.4, 0.22 * len(order) + 1.0)))
    panel.barh(np.arange(len(order)), importance[order], color=_colours(colors)[0])
    panel.set_yticks(np.arange(len(order)), [labels[position] for position in order])
    panel.invert_yaxis()
    panel.set_xlabel("Feature importance")
    return panel


def _abbreviation(label: str) -> str:
    """Short form of the name of a fuzzy set: initials of a phrase, the rank of a level."""
    words = label.split()
    if len(words) >= 2 and words[0] == "level":
        return words[1]
    return "".join(word[0].upper() for word in words)


@_shown
def plot_rules(
    estimator: Any,
    features: Sequence[int | str] | None = None,
    *,
    max_features: int = 12,
    set_names: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    cmap: str | None = None,
    ax: Any = None,
) -> Any:
    """Plot the rule base as a table: one row per rule, one column per feature.

    Each cell is the fuzzy set that the rule uses on the feature. A set has the colour it
    has in :func:`plot_memberships`, the first colour for the lowest set of a feature, and
    in small rule bases it is labelled with the initials of its name: ``VL`` for *very
    low*, ``M`` for *medium*. A blank cell is a feature that the rule does not use.

    Args:
        estimator: A fitted estimator.
        features: Features to show, as names or as column indices of the data seen in
            ``fit``. ``None`` shows every feature, or the ``max_features`` most important
            ones when the model has more.
        max_features: Number of features shown when ``features`` is ``None``.
        set_names: Names of the fuzzy sets of a feature, as in :func:`plot_memberships`.
        colors: Colours of the fuzzy sets, from the lowest to the highest. ``None`` uses the
            theme (see :func:`set_plot_theme`).
        cmap: Colormap that shades the cells from the lowest set to the highest, in place
            of one colour per set. It is also what is used, with the sequential colormap
            of the theme, when a feature has more sets than there are colours.
        ax: Axes to draw on. ``None`` creates a new figure.
        show: Display the figure and return nothing (``True``), or return it without
            displaying (``False``). ``None`` displays in an interactive session or a
            notebook, and returns it in a script or when ``ax`` is given.

    Returns:
        The ``Axes`` with the table.
    """
    check_is_fitted(estimator, "model_")
    positions = _select_features(estimator, features, max_features)
    labels = _feature_labels(estimator)
    model_names = [str(name) for name in estimator.model_.input_names]
    mf_params = estimator.get_mf_params()
    table = estimator.model_.get_rule_table()

    levels = np.full((len(table), len(positions)), np.nan)  # position of the set among those of its feature
    n_sets = 1
    text: list[list[str]] = [["" for _ in positions] for _ in table]
    for column, position in enumerate(positions):
        name = model_names[position]
        names = _set_labels(mf_params[name], set_names)
        located = (j for j, label in enumerate(names) if label is not None)
        order = sorted(located, key=lambda j: _set_position(mf_params[name][j]))
        rank = {j: i for i, j in enumerate(order)}
        n_sets = max(n_sets, len(order))
        for row, rule in enumerate(table):
            j = int(rule[name])
            if j in rank:
                levels[row, column] = rank[j]
                text[row][column] = _abbreviation(str(names[j]))

    size = (max(3.6, 0.62 * len(positions) + 1.6), max(2.4, 0.3 * len(table) + 1.4))
    _, (panel,) = _axes(ax, 1, ncols=1, panel_size=size)
    palette = _colours(colors)
    if cmap is None and n_sets <= len(palette):
        from matplotlib.colors import ListedColormap

        shades: Any = ListedColormap(list(palette[:n_sets]), name="fuzzy sets")
        image = panel.imshow(levels, cmap=shades, aspect="auto", vmin=-0.5, vmax=n_sets - 0.5)
    else:
        image = panel.imshow(levels, cmap=cmap or _theme.sequential, aspect="auto", vmin=0.0, vmax=max(n_sets - 1, 1))
    if levels.size <= _RULE_TABLE_MAX_ANNOTATED_CELLS:
        for (row, column), value in np.ndenumerate(levels):
            if not np.isnan(value):
                colour = _text_colour(image, float(value))
                panel.text(column, row, text[row][column], ha="center", va="center", color=colour, fontsize="small")
    panel.set_xticks(np.arange(len(positions)), [labels[p] for p in positions], rotation=60, ha="right")
    if len(table) <= 30:
        panel.set_yticks(np.arange(len(table)), [f"Rule {rule['rule_id']}" for rule in table])
    else:
        panel.set_ylabel("Rule")
    panel.set_title("Fuzzy set of each feature in each rule")
    return panel


def _gate_thresholds(history: dict[str, Any]) -> tuple[float | None, float | None]:
    """Thresholds that the feature and rule gates passed, when ``fit`` recorded them."""
    record = history.get("threshold", history)
    feature, rule = record.get("tau_lambda"), record.get("tau_theta")
    return (None if feature is None else float(feature), None if rule is None else float(rule))


@_shown
def plot_gates(estimator: Any, *, colors: Sequence[str] | None = None, ax: Any = None) -> Any:
    """Plot the gates of the features and of the rules that a gated family kept.

    Only DG-TSK, DG-ALETSK and FSRE-ADATSK have gates. The bars are the gate values at the
    end of the phase that trained them, for the features in ``selected_features_`` and for
    the rules of the fitted model, and the dashed line is the threshold they passed. The
    gate of FSRE-ADATSK is an odd function, so its magnitude is drawn.

    Args:
        estimator: A fitted estimator of a gated family.
        colors: Colour of the bars, as a sequence whose first colour is used. ``None`` uses
            the theme (see :func:`set_plot_theme`).
        ax: Two axes to draw on, for the features and for the rules. ``None`` creates a
            new figure.
        show: Display the figure and return nothing (``True``), or return it without
            displaying (``False``). ``None`` displays in an interactive session or a
            notebook, and returns it in a script or when ``ax`` is given.

    Returns:
        The ``Figure`` with the two panels.
    """
    check_is_fitted(estimator, "model_")
    feature_gates, rule_gates = estimator.get_feature_gates(), estimator.get_rule_gates()
    if feature_gates is None or rule_gates is None:
        raise ValueError("only DG-TSK, DG-ALETSK and FSRE-ADATSK have gates to plot")
    thresholds = _gate_thresholds(estimator.history_)
    panels = (
        ("Feature gates", np.abs(np.asarray(feature_gates, dtype=float)).ravel(), _feature_labels(estimator)),
        ("Rule gates", np.abs(np.asarray(rule_gates, dtype=float)).ravel(), None),
    )

    fig, axes = _axes(ax, 2, ncols=2, panel_size=(4.4, 3.0))
    for panel, (title, values, labels), threshold in zip(axes, panels, thresholds, strict=True):
        positions = np.arange(len(values))
        panel.bar(positions, values, color=_colours(colors)[0])
        if threshold is not None:
            panel.axhline(threshold, color=_theme.reference, linestyle="--")
            title = f"{title} (dashed: threshold {threshold:.2f})"
        if labels is None:
            labels = [f"Rule {rule}" for rule in positions]
        if len(values) <= 30:
            slanted = len(values) > 4
            panel.set_xticks(positions, labels, rotation=60 if slanted else 0, ha="right" if slanted else "center")
        else:
            panel.set_xticks([])
        panel.set_ylim(0.0, 1.1 * max(float(values.max()), threshold or 0.0))
        panel.set_ylabel("Gate value")
        panel.set_title(title)
    return fig


_PLOTTERS: dict[str, Callable[..., Any]] = {
    "memberships": plot_memberships,
    "history": plot_history,
    "rules": plot_rules,
    "rule_activation": plot_rule_activation,
    "diagnostics": plot_diagnostics,
    "feature_importance": plot_feature_importance,
    "gates": plot_gates,
}


def plot(estimator: Any, kind: str = "memberships", **kwargs: Any) -> Any:
    """Draw the plot named *kind* for a fitted estimator.

    Args:
        estimator: A fitted estimator.
        kind: One of ``"memberships"`` (default), ``"history"``, ``"rules"``,
            ``"rule_activation"``, ``"diagnostics"``, ``"feature_importance"`` or ``"gates"``.
        **kwargs: Arguments of the corresponding ``plot_<kind>`` function, among them
            ``show``.

    Returns:
        The ``Axes`` or ``Figure`` returned by that function, or ``None`` when the figure
        was displayed.
    """
    if kind not in _PLOTTERS:
        raise ValueError(f"kind must be one of {list(PLOT_KINDS)}; got {kind!r}")
    return _PLOTTERS[kind](estimator, **kwargs)


__all__: list[str] = [
    "PLOT_KINDS",
    "PlotTheme",
    "get_plot_theme",
    "plot",
    "plot_diagnostics",
    "plot_feature_importance",
    "plot_gates",
    "plot_history",
    "plot_memberships",
    "plot_rule_activation",
    "plot_rules",
    "reset_plot_theme",
    "set_plot_theme",
]
