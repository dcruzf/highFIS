"""Rules as text and the estimator-level accessors that replace ``estimator.model_...``."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

import highfis
from highfis._describe import set_labels

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

    line = clf.rules_as_text(top_features=None).split("\n")[3]

    assert line.startswith("Rule 1: IF ")
    assert f"dose is {expected}" in line


def test_consequents_show_the_model_coefficients(clf: Any) -> None:
    weights, bias = clf.get_consequent_weights(), clf.get_consequent_bias()

    first_class = clf.rules_as_text(top_features=None, digits=4).split("\n")[1]

    assert f"{bias[0, 0]:.4f}" in first_class
    for d, name in enumerate(FEATURES):
        assert f"{abs(weights[0, 0, d]):.4f} * {name}" in first_class


def test_top_features_truncates_conditions_and_coefficients(clf: Any) -> None:
    text = clf.rules_as_text(top_features=2)

    condition, consequent = text.split("\n")[:2]
    assert condition.count(" is ") == 2
    assert condition.endswith("... (2 more)")
    assert consequent.count(" * ") == 2
    assert consequent.endswith("+ ...")
    kept = np.argsort(-clf.feature_importance())[:2]
    assert all(f"{FEATURES[d]} is " in condition for d in kept)


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
    assert ((feature_gates >= 0.0) & (feature_gates <= 1.0)).all()
    assert ((rule_gates >= 0.0) & (rule_gates <= 1.0)).all()


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
    ],
)
def test_accessors_need_a_fitted_estimator(call: Any) -> None:
    with pytest.raises(NotFittedError):
        call(highfis.TSKClassifier())
