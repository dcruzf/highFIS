"""The descriptions of a fitted estimator are texts that carry their data."""

from __future__ import annotations

import numpy as np
import pytest

import highfis
from highfis import _describe as describe
from highfis._describe import Report


def _fit() -> tuple[highfis.TSKClassifier, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((80, 4))
    y = (x[:, 0] > 0.5).astype(int)
    return highfis.TSKClassifier(n_mfs=3, epochs=10, random_state=0).fit(x, y), x


def test_a_report_is_a_text_that_displays_itself() -> None:
    report = Report("first line\nsecond line", {"n": 2})

    assert isinstance(report, str)
    assert report == "first line\nsecond line"
    assert repr(report) == "first line\nsecond line"  # what a prompt or a notebook shows
    assert report.splitlines()[1] == "second line"
    assert report.to_dict() == {"n": 2}


def test_summary_carries_the_structure_of_the_model() -> None:
    est, _ = _fit()
    report = est.summary()

    assert report.startswith("TSKClassifier")
    assert report.data["task"] == "classification"
    assert report.data["n_rules"] == est.model_.n_rules
    assert report.data["n_features_in"] == 4
    counts = report.data["n_parameters"]
    assert counts["total"] == counts["antecedent"] + counts["consequent"]
    assert report.data["phases"][0]["epochs"] == 10
    assert report.data["phases"][0]["last_loss"] == est.history_["train_loss"][-1]


def test_explain_carries_the_weights_and_the_contributions() -> None:
    est, x = _fit()
    report = est.explain(x[0])

    assert report.startswith("Prediction: ")
    assert report.data["prediction"] == est.predict(x[:1])[0]
    np.testing.assert_allclose(report.data["weights"].sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(report.data["contributions"].sum(axis=0), report.data["output"])
    assert report.data["contributions"].shape == (est.model_.n_rules, 2)


def test_rules_as_text_carries_the_rule_base() -> None:
    est, _ = _fit()
    report = est.rules_as_text()

    assert report.startswith("Rule 0: IF ")
    assert report.data["features"] == ["x1", "x2", "x3", "x4"]
    assert len(report.data["rules"]) == est.model_.n_rules
    assert set(report.data["rules"][0].values()) <= {"low", "medium", "high"}
    assert report.data["weights"].shape == (est.model_.n_rules, 2, 4)


def test_a_script_sees_the_text_without_printing_it(capsys: pytest.CaptureFixture[str]) -> None:
    """Run as a script, a description prints itself, so the call is the same as in a notebook."""
    est, x = _fit()

    report = est.summary()

    assert capsys.readouterr().out == f"{report}\n"
    assert est.rules_as_text(show=False) is not None
    assert capsys.readouterr().out == ""
    est.explain(x[0], show=True)
    assert capsys.readouterr().out.startswith("Prediction: ")


def test_an_interactive_session_is_left_to_display_the_value(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    est, _ = _fit()
    monkeypatch.setattr(describe, "interactive_session", lambda: True)

    report = est.summary()

    assert capsys.readouterr().out == ""
    assert repr(report) == str(report)
