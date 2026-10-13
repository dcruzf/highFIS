"""Tests for the diagnostic plots, drawn with the non-interactive Agg backend."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import highfis  # noqa: E402
from highfis import plotting  # noqa: E402

ESTIMATORS: list[str] = sorted(n for n in highfis.__all__ if n.endswith(("Classifier", "Regressor")))


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((120, 6))
    return x, (x[:, 0] + x[:, 1] > 1.0).astype(int), 2.0 * x[:, 0] + x[:, 1]


def _fit(name: str, epochs: int = 6, **params: Any) -> Any:
    x, y, y_reg = _data()
    cls = getattr(highfis, name)
    kw = {p: epochs for p in inspect.signature(cls.__init__).parameters if p.endswith("epochs")}
    target = y if name.endswith("Classifier") else y_reg
    fit_kw: dict[str, Any] = {"metrics": ["accuracy"]} if name.endswith("Classifier") else {}
    return cls(random_state=0, **kw, **params).fit(x[:90], target[:90], x_val=x[90:], y_val=target[90:], **fit_kw)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def clf() -> Any:
    return _fit("TSKClassifier", epochs=30)


@pytest.fixture(scope="module")
def reg() -> Any:
    return _fit("TSKRegressor", epochs=30)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_every_estimator_draws_every_kind(name: str) -> None:
    x, y, y_reg = _data()
    est = _fit(name)
    target = y if name.endswith("Classifier") else y_reg

    assert isinstance(est.plot(), Figure)
    assert isinstance(est.plot(kind="history"), Figure if len(plotting._history_phases(est.history_)) > 1 else Axes)
    assert isinstance(est.plot(kind="rules"), Axes)
    assert isinstance(est.plot(kind="rule_activation", X=x[90:]), Axes)
    assert isinstance(est.plot(kind="diagnostics", X=x[90:], y=target[90:]), Figure)
    assert isinstance(est.plot(kind="feature_importance"), Axes)
    if est.get_feature_gates() is None:
        with pytest.raises(ValueError, match="have gates"):
            est.plot(kind="gates")
    else:
        assert isinstance(est.plot(kind="gates"), Figure)


def test_plot_methods_match_the_kinds(clf: Any) -> None:
    x, y, _ = _data()

    assert plotting.PLOT_KINDS == (
        "memberships",
        "history",
        "rules",
        "rule_activation",
        "diagnostics",
        "feature_importance",
        "gates",
    )
    assert isinstance(clf.plot_memberships(), Figure)
    assert isinstance(clf.plot_feature_importance(), Axes)
    assert isinstance(clf.plot_history(), Axes)
    assert isinstance(clf.plot_rules(), Axes)
    assert isinstance(clf.plot_rule_activation(x, y), Axes)
    assert isinstance(clf.plot_diagnostics(x, y), Figure)


def test_unknown_kind_is_rejected(clf: Any) -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        clf.plot(kind="roc")


def test_unfitted_estimator_is_rejected() -> None:
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        highfis.TSKClassifier().plot()


def test_missing_matplotlib_explains_how_to_install(clf: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    with pytest.raises(ImportError, match=r"pip install highfis\[plot\]"):
        clf.plot()


def test_plotting_leaves_global_state_alone(clf: Any) -> None:
    x, y, _ = _data()
    before = dict(matplotlib.rcParams)

    for kind, kwargs in (("memberships", {}), ("history", {}), ("diagnostics", {"X": x, "y": y})):
        clf.plot(kind=kind, **kwargs)

    assert dict(matplotlib.rcParams) == before


# --- memberships ------------------------------------------------------------------------


def test_memberships_one_panel_per_feature_with_one_curve_per_set(clf: Any) -> None:
    fig = clf.plot_memberships()

    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title() for ax in panels] == [f"x{i}" for i in range(1, 7)]
    assert all(len(ax.lines) == 3 for ax in panels)


def test_memberships_curves_are_the_trained_sets(clf: Any) -> None:
    """The curve must be the model's own membership function, not a redrawn formula."""
    import torch

    line = clf.plot_memberships(features=[0]).axes[0].lines[1]
    grid, drawn = line.get_xdata(), line.get_ydata()
    mf = clf.model_.membership_layer.input_mfs["x1"][1]

    with torch.no_grad():
        expected = mf(torch.as_tensor(grid, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(drawn, expected, atol=1e-6)


@pytest.mark.parametrize("mf", ["gaussian_pi", "bell", "triangular", "trapezoidal"])
def test_memberships_draw_every_selectable_shape(mf: str) -> None:
    est = _fit("TSKClassifier", mf=mf)

    panels = est.plot_memberships().axes

    assert all(len(ax.lines) == 3 for ax in panels if ax.get_visible())
    assert all(np.isfinite(line.get_ydata()).all() for ax in panels for line in ax.lines)


def test_memberships_reach_the_peak_of_a_narrow_set(clf: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A set much narrower than the grid step must still be drawn up to its centre."""
    import torch

    narrow = clf.model_.membership_layer.input_mfs["x1"][0]
    monkeypatch.setattr(type(narrow), "forward", lambda self, x: torch.exp(-((x - self.mean) ** 2) / 2e-10))

    line = clf.plot_memberships(features=[0], n_points=20).axes[0].lines[0]

    assert float(np.max(line.get_ydata())) == pytest.approx(1.0)


def test_memberships_select_features_by_index_and_name(clf: Any) -> None:
    by_index = clf.plot_memberships(features=[4, 1])
    by_name = clf.plot_memberships(features=["x5", "x2"])

    assert [ax.get_title() for ax in by_index.axes] == ["x5", "x2"]
    assert [ax.get_title() for ax in by_name.axes] == ["x5", "x2"]


def test_memberships_reject_unknown_features(clf: Any) -> None:
    with pytest.raises(ValueError, match="unknown feature"):
        clf.plot_memberships(features=["petal length"])
    with pytest.raises(ValueError, match="not used by the fitted model"):
        clf.plot_memberships(features=[17])


def test_memberships_show_the_most_important_features_when_there_are_many(clf: Any) -> None:
    fig = clf.plot_memberships(max_features=2)

    top = sorted(np.argsort(-clf.feature_importance())[:2].tolist())
    assert [ax.get_title() for ax in fig.axes if ax.get_visible()] == [f"x{i + 1}" for i in top]


def test_memberships_use_the_feature_names_seen_in_fit(clf: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    names = np.array(["age", "dose", "weight", "height", "bmi", "score"], dtype=object)
    monkeypatch.setattr(clf, "feature_names_in_", names, raising=False)

    fig = clf.plot_memberships(features=["dose", 5])

    assert [ax.get_title() for ax in fig.axes] == ["dose", "score"]


def test_memberships_span_the_data_when_given(clf: Any) -> None:
    x, _, _ = _data()

    lo, hi = clf.plot_memberships(features=[0], X=x * 4.0).axes[0].lines[0].get_xdata()[[0, -1]]

    assert lo < 0.0 < 3.9 < hi


def test_memberships_follow_pruned_models() -> None:
    """A model that dropped features shows the surviving ones under their own names."""
    est = _fit("DGTSKRegressor")
    kept = [str(n) for n in est.model_.input_names]
    assert len(kept) < 6

    fig = est.plot_memberships()

    assert [ax.get_title() for ax in fig.axes if ax.get_visible()] == kept


def test_memberships_skip_dont_care_sets() -> None:
    est = _fit("MHTSKClassifier")
    sets = est.model_.membership_layer.input_mfs["x1"]
    real = [mf for mf in sets if type(mf).__name__ != "ConstantMF"]
    assert len(real) < len(sets)

    assert len(est.plot_memberships(features=[0]).axes[0].lines) == len(real)


def test_memberships_draw_on_given_axes(clf: Any) -> None:
    fig, axes = plt.subplots(1, 2)

    out = clf.plot_memberships(features=[0, 1], ax=axes)

    assert out is fig
    assert all(len(ax.lines) == 3 for ax in axes)
    with pytest.raises(ValueError, match="needs 3 axes"):
        clf.plot_memberships(features=[0, 1, 2], ax=axes)


# --- history ----------------------------------------------------------------------------


def test_history_shows_training_validation_and_best_epoch(clf: Any) -> None:
    ax = clf.plot_history()

    labels = [line.get_label() for line in ax.lines]
    assert labels[:2] == ["training", "validation"]
    assert "best epoch" in labels
    assert ax.get_ylabel() == "Loss"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), clf.history_["train_loss"])


def test_history_without_validation_has_only_the_training_curve() -> None:
    x, y, _ = _data()
    est = highfis.TSKClassifier(epochs=5, random_state=0).fit(x, y)

    assert [line.get_label() for line in est.plot_history().lines] == ["training"]


def test_history_plots_a_recorded_metric(clf: Any) -> None:
    ax = clf.plot_history("accuracy")

    assert ax.get_ylabel() == "Accuracy"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), clf.history_["train_accuracy"])
    with pytest.raises(ValueError, match="was not recorded during fit"):
        clf.plot_history("f1")


@pytest.mark.parametrize(
    ("name", "phases"),
    [
        ("DGALETSKClassifier", ["DG", "Fine-tune"]),
        ("FSREADATSKClassifier", ["Feature selection", "Rule extraction", "Fine-tune"]),
    ],
)
def test_history_has_one_panel_per_training_phase(name: str, phases: list[str]) -> None:
    est = _fit(name)

    fig = est.plot_history()

    assert isinstance(fig, Figure)
    assert [panel.get_title() for panel in fig.axes] == phases
    keys = [key for key in ("dg", "fs", "re", "finetune") if key in est.history_]
    for panel, key in zip(fig.axes, keys, strict=True):
        np.testing.assert_allclose(np.asarray(panel.lines[0].get_ydata(), dtype=float), est.history_[key]["train_loss"])
    # The panels share the vertical scale, and only the first names it.
    assert len({panel.get_ylim() for panel in fig.axes}) == 1
    assert [bool(panel.get_ylabel()) for panel in fig.axes] == [True] + [False] * (len(phases) - 1)


def test_history_of_several_phases_draws_on_given_axes() -> None:
    est = _fit("FSREADATSKClassifier")
    fig, axes = plt.subplots(1, 3)
    assert est.plot_history(ax=axes) is fig
    with pytest.raises(ValueError, match="needs 3 axes"):
        est.plot_history(ax=axes[0])


def test_history_draws_on_a_given_axes(clf: Any) -> None:
    _, ax = plt.subplots()

    assert clf.plot_history(ax=ax) is ax


# --- rule activation --------------------------------------------------------------------


def test_rule_activation_by_class_is_the_class_mean(clf: Any) -> None:
    x, y, _ = _data()

    ax = clf.plot_rule_activation(x, y)

    activation = clf.rule_activation(x)
    expected = np.stack([activation[y == c].mean(axis=0) for c in clf.classes_], axis=1)
    np.testing.assert_allclose(ax.images[0].get_array(), expected)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Rule 0", "Rule 1", "Rule 2"]


def test_rule_activation_without_labels_is_a_bar_per_rule(reg: Any) -> None:
    x, _, _ = _data()

    ax = reg.plot_rule_activation(x)

    np.testing.assert_allclose([bar.get_width() for bar in ax.patches], reg.rule_activation(x).mean(axis=0))


def test_rule_activation_rejects_labels_for_a_regressor(reg: Any) -> None:
    x, _, y_reg = _data()

    with pytest.raises(ValueError, match="only applies to classifiers"):
        reg.plot_rule_activation(x, y_reg)


def test_rule_activation_handles_many_rules() -> None:
    x, y, _ = _data()
    est = _fit("MHTSKClassifier")
    assert est.model_.n_rules > 30

    ax = est.plot_rule_activation(x, y)

    assert ax.get_ylabel() == "Rule"


# --- diagnostics ------------------------------------------------------------------------


def test_classifier_diagnostics_show_confusion_matrix_and_margin(clf: Any) -> None:
    from sklearn.metrics import confusion_matrix

    x, y, _ = _data()

    matrix_ax, margin_ax = clf.plot_diagnostics(x, y).axes

    np.testing.assert_array_equal(matrix_ax.images[0].get_array(), confusion_matrix(y, clf.predict(x)))
    margins = np.concatenate([c.get_offsets()[:, 1] for c in margin_ax.collections])
    assert len(margins) == len(y)
    # A negative margin is a misclassified sample, so the two panels must agree.
    assert int((margins < 0).sum()) == int((clf.predict(x) != y).sum())


def test_classifier_diagnostics_work_with_string_labels_and_three_classes() -> None:
    rng = np.random.default_rng(1)
    x = rng.random((90, 4))
    y = np.array(["low", "mid", "high"])[np.digitize(x[:, 0], [0.33, 0.66])]
    est = highfis.TSKClassifier(epochs=10, random_state=0).fit(x, y)

    matrix_ax, margin_ax = est.plot_diagnostics(x, y).axes

    assert [t.get_text() for t in matrix_ax.get_xticklabels()] == ["high", "low", "mid"]
    assert sum(len(c.get_offsets()) for c in margin_ax.collections) == 90


def test_regressor_diagnostics_show_fit_and_residuals(reg: Any) -> None:
    x, _, y_reg = _data()

    fit_ax, residual_ax = reg.plot_diagnostics(x, y_reg).axes

    predicted = reg.predict(x)
    np.testing.assert_allclose(fit_ax.collections[0].get_offsets(), np.column_stack([y_reg, predicted]))
    np.testing.assert_allclose(residual_ax.collections[0].get_offsets()[:, 1], predicted - y_reg, atol=1e-6)


def test_diagnostics_draw_on_given_axes(clf: Any) -> None:
    x, y, _ = _data()
    fig, axes = plt.subplots(1, 2)

    assert clf.plot_diagnostics(x, y, ax=axes) is fig


# --- feature importance and gates ---------------------------------------------------------


def test_feature_importance_draws_the_largest_values_in_order(clf: Any) -> None:
    importance = clf.feature_importance()
    panel = clf.plot_feature_importance(top=3)

    widths = [bar.get_width() for bar in panel.patches]
    assert widths == pytest.approx(np.sort(importance)[::-1][:3].tolist())
    order = np.argsort(-importance)[:3]
    assert [tick.get_text() for tick in panel.get_yticklabels()] == [clf.model_.input_names[i] for i in order]


def test_feature_importance_draws_every_feature_when_asked(clf: Any) -> None:
    assert len(clf.plot_feature_importance(top=None).patches) == clf.model_.n_inputs


@pytest.mark.parametrize("name", ["DGTSKClassifier", "DGALETSKClassifier", "FSREADATSKClassifier"])
def test_gates_show_the_kept_features_and_rules_with_their_thresholds(name: str) -> None:
    est = _fit(name, epochs=40)
    features_panel, rules_panel = est.plot_gates().axes

    assert len(features_panel.patches) == len(est.selected_features_)
    assert len(rules_panel.patches) == est.model_.n_rules
    heights = [bar.get_height() for bar in features_panel.patches]
    assert heights == pytest.approx(np.abs(est.get_feature_gates()).tolist())
    for panel in (features_panel, rules_panel):
        assert any(line.get_linestyle() == "--" for line in panel.lines)


def test_gates_are_rejected_for_a_family_without_gates(clf: Any) -> None:
    with pytest.raises(ValueError, match="have gates"):
        clf.plot_gates()


def test_gates_draw_on_given_axes() -> None:
    est = _fit("DGTSKClassifier", epochs=40)
    fig, axes = plt.subplots(1, 2)

    assert est.plot_gates(ax=axes) is fig


def test_memberships_are_named_as_in_the_rules(clf: Any) -> None:
    """The legend speaks of low, medium and high, as ``rules_as_text`` does, and sits beside the panels."""
    fig = clf.plot_memberships()

    assert [text.get_text() for text in fig.legends[0].get_texts()] == ["low", "medium", "high"]
    assert all(panel.get_legend() is None for panel in fig.axes)


def test_memberships_take_other_set_names(clf: Any) -> None:
    fig = clf.plot_memberships(set_names=["cold", "mild", "hot"])
    assert [text.get_text() for text in fig.legends[0].get_texts()] == ["cold", "mild", "hot"]


def test_memberships_on_given_axes_keep_a_legend_per_panel(clf: Any) -> None:
    _, axes = plt.subplots(1, 6)
    fig = clf.plot_memberships(ax=axes)
    assert not fig.legends
    assert all(panel.get_legend() is not None for panel in axes)


def test_rules_table_has_the_set_of_each_rule_and_feature(clf: Any) -> None:
    panel = clf.plot_rules()
    levels = np.asarray(panel.images[0].get_array())
    table = clf.model_.get_rule_table()

    assert levels.shape == (len(table), 6)
    # Each feature has three sets, and each rule of the compact rule base uses one of them.
    assert all(sorted(column) == [0.0, 1.0, 2.0] for column in levels.T.tolist())
    first = clf.get_mf_params()[clf.model_.input_names[0]]
    lowest = int(np.argmin([params["mean"] for params in first]))
    assert levels[lowest, 0] == 0.0
    assert sorted({text.get_text() for text in panel.texts}) == ["H", "L", "M"]
    assert [tick.get_text() for tick in panel.get_yticklabels()] == ["Rule 0", "Rule 1", "Rule 2"]


def test_rules_table_shows_the_chosen_features(clf: Any) -> None:
    panel = clf.plot_rules(features=[0, 3])
    assert np.asarray(panel.images[0].get_array()).shape[1] == 2
    assert [tick.get_text() for tick in panel.get_xticklabels()] == ["x1", "x4"]


def test_gate_thresholds_are_in_the_titles() -> None:
    est = _fit("FSREADATSKClassifier", epochs=30)
    fig = est.plot_gates()
    for panel in fig.axes[:2]:
        assert "threshold" in panel.get_title()
        assert panel.get_legend() is None


# --- theme ------------------------------------------------------------------------------


@pytest.fixture
def _theme() -> Iterator[None]:
    yield
    highfis.reset_plot_theme()


def _hex(colour: Any) -> str:
    return matplotlib.colors.to_hex(colour)


def test_every_plot_uses_the_colours_of_the_theme(clf: Any) -> None:
    x, y, _ = _data()
    theme = highfis.get_plot_theme()

    assert [_hex(line.get_color()) for line in clf.plot_history().lines[:2]] == [_hex(c) for c in theme.colors[:2]]
    assert _hex(clf.plot_feature_importance().patches[0].get_facecolor()) == _hex(theme.colors[0])
    # The table of the rule base gives each fuzzy set the colour it has in the membership plot.
    table = clf.plot_rules().images[0]
    assert [_hex(table.cmap(table.norm(rank))) for rank in range(3)] == [_hex(c) for c in theme.colors[:3]]
    curves = clf.plot_memberships().axes[0].lines
    assert sorted(_hex(line.get_color()) for line in curves) == sorted(_hex(c) for c in theme.colors[:3])
    assert clf.plot_rule_activation(x, y).images[0].get_cmap().name == theme.sequential
    assert clf.plot_diagnostics(x, y).axes[0].images[0].get_cmap().name == theme.sequential


def test_the_theme_changes_the_colours_of_the_next_plots(clf: Any, _theme: None) -> None:
    x, y, _ = _data()
    theme = highfis.set_plot_theme(colors=["black", "red"], sequential="Greys")

    assert theme.colors == ("black", "red")
    assert theme.reference == "0.4"  # what is not given is kept
    assert [_hex(line.get_color()) for line in clf.plot_history().lines[:2]] == ["#000000", "#ff0000"]
    # Three fuzzy sets and two colours: the table is shaded with the sequential colormap.
    assert clf.plot_rules().images[0].get_cmap().name == "Greys"
    assert clf.plot_rule_activation(x, y).images[0].get_cmap().name == "Greys"

    assert highfis.reset_plot_theme() == plotting.PlotTheme()
    assert clf.plot_rules().images[0].get_cmap().name == "fuzzy sets"


def test_a_call_can_choose_its_own_colours(clf: Any) -> None:
    x, y, _ = _data()

    assert clf.plot_rules(cmap="viridis").images[0].get_cmap().name == "viridis"
    table = clf.plot_rules(colors=["white", "grey", "black"]).images[0]
    assert [_hex(table.cmap(table.norm(rank))) for rank in range(3)] == ["#ffffff", "#808080", "#000000"]
    assert clf.plot_diagnostics(x, y, cmap="Greys").axes[0].images[0].get_cmap().name == "Greys"
    assert _hex(clf.plot_feature_importance(colors=["green"]).patches[0].get_facecolor()) == "#008000"
    lines = clf.plot_memberships(colors=["black", "red", "blue"]).axes[0].lines
    assert sorted(_hex(line.get_color()) for line in lines) == ["#000000", "#0000ff", "#ff0000"]
    assert highfis.get_plot_theme() == plotting.PlotTheme()  # the theme is left as it was


def test_an_empty_list_of_colours_is_rejected(clf: Any, _theme: None) -> None:
    with pytest.raises(ValueError, match="at least one colour"):
        highfis.set_plot_theme(colors=[])
    with pytest.raises(ValueError, match="at least one colour"):
        clf.plot_history(colors=[])


def test_heatmap_labels_contrast_with_their_cells(clf: Any) -> None:
    panel = clf.plot_rules(cmap="Greys")
    image = panel.images[0]
    levels = np.asarray(image.get_array())
    for text in panel.texts:
        column, row = (round(v) for v in text.get_position())
        red, green, blue, _ = image.cmap(image.norm(levels[row, column]))
        dark = 0.299 * red + 0.587 * green + 0.114 * blue < 0.5
        assert text.get_color() == ("white" if dark else "black")


# --- show -------------------------------------------------------------------------------


def test_a_script_gets_the_figure_and_no_window(clf: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(True))
    monkeypatch.setattr(plotting, "_interactive_session", lambda: False)

    assert isinstance(clf.plot("history"), Axes)
    assert not shown


def test_an_interactive_session_sees_the_figure_and_gets_nothing(clf: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(True))
    monkeypatch.setattr(plotting, "_interactive_session", lambda: True)

    assert clf.plot("history") is None
    assert clf.plot_rules() is None
    assert shown == [True, True]
    # Asking for the figure, or drawing on one's own axes, shows nothing.
    assert isinstance(clf.plot("history", show=False), Axes)
    _, ax = plt.subplots()
    assert clf.plot_history(ax=ax) is ax
    assert shown == [True, True]


def test_show_can_be_forced_in_a_script(clf: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(True))
    monkeypatch.setattr(plotting, "_interactive_session", lambda: False)

    assert clf.plot("memberships", show=True) is None
    assert shown == [True]


def test_a_rule_table_that_cannot_show_every_feature_shows_those_that_tell_the_rules_apart() -> None:
    """The features on which a rule departs from its usual fuzzy set are shown first."""

    class Model:
        def feature_importance(self) -> np.ndarray:
            return np.array([0.4, 0.3, 0.2, 0.05, 0.05])

    names = ["a", "b", "c", "d", "e"]
    table = [
        dict(zip(names, [0, 0, 0, 0, 1], strict=True)),  # e is the exception
        dict(zip(names, [0, 0, 0, 1, 0], strict=True)),  # d is the exception
        dict(zip(names, [1, 1, 1, 1, 0], strict=True)),  # e again
    ]

    assert plotting._distinguishing_features(Model(), table, names, 1) == [4]
    assert plotting._distinguishing_features(Model(), table, names, 2) == [3, 4]
    # Once the exceptions are in, the most important features complete the selection.
    assert plotting._distinguishing_features(Model(), table, names, 3) == [0, 3, 4]
