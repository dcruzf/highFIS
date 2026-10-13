"""Rules as text and the estimator-level accessors that replace ``estimator.model_...``."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

import highfis
from highfis import _describe as describe
from highfis._describe import _conditions, set_labels

ESTIMATORS: list[str] = sorted(n for n in highfis.__all__ if n.endswith(("Classifier", "Regressor")))
GATED = {n for n in ESTIMATORS if n.startswith(("DGTSK", "DGALETSK", "FSREADATSK"))}
FEATURES = ["age", "dose", "weight", "height"]


def _fit(name: str, n_features: int = 6, **params: Any) -> Any:
    rng = np.random.default_rng(0)
    x = rng.random((80, n_features))
    y: Any = (x[:, 0] > 0.5).astype(int) if name.endswith("Classifier") else 2.0 * x[:, 0]
    cls = getattr(highfis, name)
    epochs = {p: 3 for p in inspect.signature(cls.__init__).parameters if p.endswith("epochs")}
    return cls(random_state=0, **{**epochs, **params}).fit(x, y)


@pytest.fixture(scope="module")
def clf() -> Any:
    rng = np.random.default_rng(0)
    x = rng.random((90, 4))
    y = np.array(["low risk", "high risk"])[(x[:, 1] + x[:, 2] > 1.0).astype(int)]
    est = highfis.TSKClassifier(n_mfs=3, epochs=20, random_state=0).fit(x, y)
    est.feature_names_in_ = np.array(FEATURES, dtype=object)
    return est


# --- naming the fuzzy sets --------------------------------------------------------------


@pytest.mark.parametrize(
    ("centres", "expected"),
    [
        ([0.9, 0.1], ["high", "low"]),
        ([0.5, 0.9, 0.1], ["medium", "high", "low"]),
        ([0.1, 0.3, 0.5, 0.7, 0.9], ["very low", "low", "medium", "high", "very high"]),
        ([0.4, 0.1, 0.9, 0.6], ["level 2 of 4", "level 1 of 4", "level 4 of 4", "level 3 of 4"]),
    ],
)
def test_sets_are_named_from_the_order_of_their_centres(centres: list[float], expected: list[str]) -> None:
    assert set_labels([{"mean": c, "sigma": 0.1} for c in centres]) == expected


def test_set_names_can_be_replaced_when_the_count_matches() -> None:
    params = [{"center": 0.8}, {"center": 0.2}]

    assert set_labels(params, ["cold", "hot"]) == ["hot", "cold"]
    assert set_labels(params, ["cold", "warm", "hot"]) == ["high", "low"]


def test_sets_without_a_location_are_not_named() -> None:
    params = [{"value": 1.0}, {"mean": 0.7, "sigma": 0.1}, {"a": 0.0, "b": 0.1, "c": 0.3, "d": 0.4}]

    assert set_labels(params) == [None, "high", "low"]


# --- rules as text ----------------------------------------------------------------------


def test_classifier_rules_use_feature_and_class_names(clf: Any) -> None:
    text = clf.rules_as_text(top_features=None)

    blocks = text.split("\nRule ")
    assert len(blocks) == clf.model_.n_rules
    assert text.startswith("Rule 0: IF ")
    assert all(f"{name} is " in blocks[0] for name in FEATURES)
    assert "THEN high risk: " in blocks[0]
    assert "     low risk: " in blocks[0]
    assert "..." not in text


def test_rule_conditions_follow_the_rule_table(clf: Any) -> None:
    """The set named in a condition is the one the rule table points to."""
    rule = clf.get_rule_table()[1]
    params = clf.get_mf_params()["x2"]
    expected = set_labels(params)[rule["x2"]]

    line = next(line for line in clf.rules_as_text(top_features=None).split("\n") if line.startswith("Rule 1: IF "))

    assert f"dose is {expected}" in line


def test_consequents_show_the_model_coefficients(clf: Any) -> None:
    weights, bias = clf.get_consequent_weights(), clf.get_consequent_bias()

    first_class = clf.rules_as_text(top_features=None, digits=4, width=200).split("\n")[1]

    assert f"{bias[0, 0]:.4f}" in first_class
    for d, name in enumerate(FEATURES):
        assert f"{abs(weights[0, 0, d]):.4f} * {name}" in first_class


def test_top_features_truncates_the_coefficients_and_not_the_conditions(clf: Any) -> None:
    text = clf.rules_as_text(top_features=2)

    condition, consequent = text.split("\n")[:2]
    assert all(f"{feature} is " in condition for feature in FEATURES)
    assert consequent.count(" * ") == 2
    assert consequent.endswith("+ ...")


def test_long_rules_are_continued_between_conditions_and_terms(clf: Any) -> None:
    narrow = clf.rules_as_text(top_features=None, width=40)

    assert max(len(line) for line in narrow.splitlines()) <= 40
    # Joined again, the lines give the text written without a limit.
    wide = clf.rules_as_text(top_features=None, width=10_000)
    assert " ".join(narrow.split()) == " ".join(wide.split())


def test_a_rule_is_written_as_what_most_features_share_and_the_exceptions() -> None:
    names = ["a", "b", "c", "d", "e"]
    sets: dict[str, list[str | None]] = {name: ["low", "high"] for name in names}

    def conditions(*indices: int) -> list[str]:
        return _conditions(dict(zip(names, indices, strict=True)), names, names, sets)

    assert conditions(0, 0, 0, 0, 0) == ["all 5 features are low"]
    assert conditions(0, 1, 0, 0, 0) == ["b is high", "the other 4 features are low"]
    assert conditions(1, 1, 1, 0, 1) == ["d is low", "the other 4 features are high"]
    # Without a set used by more than half of the features, every condition is written.
    assert _conditions({"a": 0, "b": 1}, ["a", "b"], ["a", "b"], sets) == ["a is low", "b is high"]


def test_largest_coefficients_come_first(clf: Any) -> None:
    weights = clf.get_consequent_weights()[0, 0]

    consequent = clf.rules_as_text(top_features=1).split("\n")[1]

    assert f"* {FEATURES[int(np.argmax(np.abs(weights)))]}" in consequent


def test_custom_labels_and_max_rules(clf: Any) -> None:
    text = clf.rules_as_text(labels=["small", "average", "large"], max_rules=1)

    assert " is small" in text or " is average" in text or " is large" in text
    assert " is low" not in text
    assert text.endswith("... (2 more rules)")
    assert text.count("Rule ") == 1


def test_regressor_rules_have_one_consequent() -> None:
    reg = _fit("HTSKRegressor", n_features=3, n_mfs=2)

    lines = reg.rules_as_text().split("\n")

    assert len(lines) == 2 * reg.model_.n_rules
    assert lines[0].startswith("Rule 0: IF x1 is ")
    assert lines[1].strip().startswith("THEN y = ")


def test_dont_care_sets_are_left_out_of_the_conditions() -> None:
    est = _fit("MHTSKClassifier")
    table = est.get_rule_table()[0]
    params = est.get_mf_params()
    active = [name for name in params if params[name][table[name]]["type"] != "ConstantMF"]
    assert 0 < len(active) < 6

    condition = est.rules_as_text(top_features=None).split("\n")[0]

    assert condition.count(" is ") == len(active)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_every_estimator_describes_its_rules(name: str) -> None:
    est = _fit(name)

    text = est.rules_as_text(max_rules=2)

    assert text.startswith("Rule 0: IF ")
    assert "THEN " in text
    assert "nan" not in text


# --- accessors --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ESTIMATORS)
def test_accessors_describe_the_fitted_model(name: str) -> None:
    est = _fit(name)
    n_rules, n_inputs = est.model_.n_rules, est.model_.n_inputs

    assert len(est.get_rule_table()) == n_rules
    assert est.get_consequent_weights().shape[0] == n_rules
    assert est.get_consequent_weights().shape[-1] == n_inputs
    assert est.get_consequent_bias().shape[0] == n_rules
    assert est.selected_features_.shape == (n_inputs,)
    assert isinstance(est.get_consequent_weights(), np.ndarray)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_gates_exist_only_in_the_gated_families(name: str) -> None:
    est = _fit(name)

    feature_gates, rule_gates = est.get_feature_gates(), est.get_rule_gates()

    if name not in GATED:
        assert feature_gates is None
        assert rule_gates is None
        return
    assert feature_gates.shape == (est.model_.n_inputs,)
    assert rule_gates.shape == (est.model_.n_rules,)
    # The gate of FSRE-ADATSK is an odd function with values in [-1, 1]; the others are in [0, 1].
    lowest = -1.0 if name.startswith("FSREADATSK") else 0.0
    assert ((feature_gates >= lowest) & (feature_gates <= 1.0)).all()
    assert ((rule_gates >= lowest) & (rule_gates <= 1.0)).all()


def test_selected_features_are_every_column_without_pruning(clf: Any) -> None:
    np.testing.assert_array_equal(clf.selected_features_, [0, 1, 2, 3])


def test_selected_features_are_the_survivors_after_pruning() -> None:
    est = _fit("FSREADATSKRegressor")
    survivors = est.selected_features_
    assert len(survivors) < 6

    rng = np.random.default_rng(1)
    x = rng.random((5, 6))
    reduced = est._select_model_features(x)

    np.testing.assert_array_equal(reduced, x[:, survivors])
    assert [f"x{i + 1}" for i in survivors] == [str(n) for n in est.model_.input_names]


@pytest.mark.parametrize(
    "call",
    [
        lambda e: e.rules_as_text(),
        lambda e: e.get_rule_table(),
        lambda e: e.get_consequent_weights(),
        lambda e: e.get_consequent_bias(),
        lambda e: e.get_feature_gates(),
        lambda e: e.get_rule_gates(),
        lambda e: e.selected_features_,
        lambda e: e.summary(),
        lambda e: e.explain(np.zeros(4)),
        lambda e: e.coverage(np.zeros((1, 4))),
    ],
)
def test_accessors_need_a_fitted_estimator(call: Any) -> None:
    with pytest.raises(NotFittedError):
        call(highfis.TSKClassifier())


# --- summary, explain and coverage --------------------------------------------------------


def _data(name: str, n_features: int = 6) -> tuple[np.ndarray, Any]:
    rng = np.random.default_rng(0)
    x = rng.random((80, n_features))
    return x, ((x[:, 0] > 0.5).astype(int) if name.endswith("Classifier") else 2.0 * x[:, 0])


@pytest.mark.parametrize("name", ESTIMATORS)
def test_every_estimator_has_a_summary(name: str) -> None:
    est = _fit(name)
    text = est.summary()

    assert text.splitlines()[0] == name
    assert f"{est.model_.n_rules} (rule base" in text
    assert f"6 seen in fit, {est.model_.n_inputs} used by the model" in text
    assert "Training" in text and "epochs" in text


def test_summary_reports_what_fit_resolved(clf: Any) -> None:
    text = clf.summary(digits=2)

    assert "classification, 2 classes" in text
    assert "3 per feature, GaussianMF" in text
    header, row = text.splitlines()[-2:]
    assert header.split() == ["phase", "epochs", "rate", "loss", "first", "->", "last", "accuracy"]
    assert row.split()[:3] == ["training", "20", f"{clf.learning_rate_:g}"]
    assert max(len(line) for line in text.splitlines()) <= 80
    trainable = sum(p.numel() for p in clf.model_.parameters() if p.requires_grad)
    assert f"{trainable} trainable" in text


def test_summary_lists_the_phases_of_a_gated_family() -> None:
    text = _fit("FSREADATSKClassifier").summary()

    assert all(phase in text for phase in ("feature selection", "rule extraction", "fine-tuning"))
    assert text.index("feature selection") < text.index("rule extraction") < text.index("fine-tuning")


@pytest.mark.parametrize("name", ESTIMATORS)
def test_rule_contributions_add_up_to_the_output(name: str) -> None:
    import torch

    est = _fit(name)
    x, _ = _data(name)
    row = x[:1]
    weights, contributions = est._rule_contributions(row)

    assert weights.shape == (est.model_.n_rules,)
    np.testing.assert_allclose(weights, est.rule_activation(row)[0], rtol=1e-5, atol=1e-6)
    was_training = est.model_.training
    est.model_.eval()
    with torch.no_grad():
        output = est.model_(est._as_tensor_x(est._select_model_features(row), torch.device("cpu"))).reshape(-1)
    est.model_.train(was_training)
    np.testing.assert_allclose(contributions.sum(axis=0), output.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_every_estimator_explains_a_sample(name: str) -> None:
    est = _fit(name)
    x, _ = _data(name)
    text = est.explain(x[0])

    assert text.startswith("Prediction: ")
    assert f"contributions of the {est.model_.n_rules} rules" in text
    assert "weight" in text and "IF " in text


def test_explain_names_classes_features_and_the_heaviest_rules(clf: Any) -> None:
    rng = np.random.default_rng(1)
    sample = rng.random(4)
    text = clf.explain(sample, top_rules=1)
    weights = clf.rule_activation(sample.reshape(1, -1))[0]

    assert text.splitlines()[0] == f"Prediction: {clf.predict(sample.reshape(1, -1))[0]}"
    assert "low risk: " in text and "high risk: " in text
    assert f"Rule {int(weights.argmax())}, weight {weights.max():.2f}" in text
    assert f"{len(weights) - 1} other rule" in text
    assert all(feature in text for feature in FEATURES)
    assert "-0.00" not in text


def test_explain_describes_every_rule_when_asked(clf: Any) -> None:
    text = clf.explain(np.full(4, 0.5), top_rules=None)

    assert text.count("Rule ") == clf.model_.n_rules
    assert "other rule" not in text


def test_explain_of_a_regressor_has_one_value_per_rule() -> None:
    est = _fit("TSKRegressor")
    x, _ = _data("TSKRegressor")
    text = est.explain(x[0], digits=3)

    assert text.splitlines()[0] == f"Prediction: {est.predict(x[:1])[0]:.3f}"


def test_explain_takes_one_sample(clf: Any) -> None:
    with pytest.raises(ValueError):
        clf.explain(np.zeros((2, 4)))


@pytest.mark.parametrize("name", ESTIMATORS)
def test_every_estimator_reports_coverage(name: str) -> None:
    est = _fit(name)
    x, _ = _data(name)
    coverage = est.coverage(x)

    assert coverage.shape == (len(x),)
    assert np.all((coverage >= 0.0) & (coverage <= 1.0 + 1e-6))


def test_coverage_falls_away_from_the_training_data(clf: Any) -> None:
    rng = np.random.default_rng(0)
    inside = clf.coverage(rng.random((50, 4)))
    outside = clf.coverage(rng.random((50, 4)) + 5.0)

    assert outside.max() < inside.min()


def test_coverage_is_one_at_the_centre_of_a_rule(clf: Any) -> None:
    params = clf.get_mf_params()
    first_rule = clf.model_.get_rule_table()[0]
    centre = [params[name][first_rule[name]]["mean"] for name in clf.model_.input_names]

    assert clf.coverage(np.asarray([centre]))[0] == pytest.approx(1.0, abs=1e-5)


def test_coverage_ignores_dont_care_sets() -> None:
    est = _fit("MHTSKClassifier", n_features=12)
    x, _ = _data("MHTSKClassifier", n_features=12)

    # A head uses a few of the features: the others must not pull the match towards one.
    assert est.coverage(x + 5.0).max() < est.coverage(x).min()


def test_summary_of_a_gated_family_fits_the_page() -> None:
    """One row per phase and one column per quantity, instead of one long sentence per phase."""
    text = _fit("FSREADATSKClassifier").summary()
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Training"))

    assert lines[start + 1].split()[:2] == ["phase", "epochs"]
    assert [line.split("  ")[1] for line in lines[start + 2 : start + 5]] == [
        "feature selection",
        "rule extraction",
        "fine-tuning",
    ]
    assert max(len(line) for line in lines) <= 80


def test_explain_groups_the_rules_that_do_not_fire() -> None:
    """A rule whose weight is written as zero is counted with the others, unless every rule is asked for."""
    est = _fit("HTSKClassifier")
    weights = np.array([0.999, 0.001, 0.0])
    contributions = np.array([[0.9, 0.1], [-0.0001, 0.0], [0.0, 0.0]])
    text = describe.explain(est, weights, contributions, est.classes_[0], top_rules=3)

    assert text.count("\nRule ") == 1
    assert "2 other rules, weight 0.00 in all" in text
    assert "-0.00" not in text
