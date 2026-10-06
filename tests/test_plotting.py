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
    assert isinstance(est.plot(kind="history"), Axes)
    assert isinstance(est.plot(kind="rule_activation", X=x[90:]), Axes)
    assert isinstance(est.plot(kind="diagnostics", X=x[90:], y=target[90:]), Figure)


def test_plot_methods_match_the_kinds(clf: Any) -> None:
    x, y, _ = _data()

    assert plotting.PLOT_KINDS == ("memberships", "history", "rule_activation", "diagnostics")
    assert isinstance(clf.plot_memberships(), Figure)
    assert isinstance(clf.plot_history(), Axes)
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
        ("DGALETSKClassifier", ["DG", "fine-tune"]),
        ("FSREADATSKClassifier", ["feature selection", "rule extraction", "fine-tune"]),
    ],
)
def test_history_lays_out_training_phases_in_order(name: str, phases: list[str]) -> None:
    est = _fit(name)

    ax = est.plot_history()

    assert [text.get_text().strip() for text in ax.texts] == phases
    total = sum(len(est.history_[key]["train_loss"]) for key in ("dg", "fs", "re", "finetune") if key in est.history_)
    curves = [line for line in ax.lines if len(line.get_xdata()) > 2]
    assert max(float(np.max(line.get_xdata())) for line in curves) == total


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
