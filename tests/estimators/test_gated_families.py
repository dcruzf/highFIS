"""Fidelity of the gated families (DG-TSK, DG-ALETSK, FSRE-ADATSK) to their source articles.

These tests pin the points where the package used to differ from the articles: how the
point-based rule base and the gates are initialized, and what happens to the gates once
the features and rules are selected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import highfis
from highfis import DGALETSKClassifier, DGALETSKRegressor, DGTSKClassifier, DGTSKRegressor, FSREADATSKClassifier
from highfis.estimators._base import _pfrb_spreads, _select_pfrb_indices

DG_ESTIMATORS = [DGTSKClassifier, DGTSKRegressor, DGALETSKClassifier, DGALETSKRegressor]


def _data(n: int = 60, d: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((n, d)) * np.linspace(0.2, 1.0, d)  # features with different spreads
    return x, (x[:, 0] > 0.1).astype(int), 2.0 * x[:, 0] + x[:, 1]


def _fit(cls: Any, **params: Any) -> tuple[Any, np.ndarray]:
    x, y, y_reg = _data()
    target = y if cls.__name__.endswith("Classifier") else y_reg
    return cls(dg_epochs=4, finetune_epochs=4, random_state=0, **params).fit(x, target), x


def _initial_sigmas(estimator: Any, x: np.ndarray) -> np.ndarray:
    input_mfs, _, _ = estimator._build_input_mfs_impl(x)
    return np.array([[float(mf.sigma.detach()) for mf in mfs] for mfs in input_mfs.values()])


# --- spreads of the point-based rule base ------------------------------------------------


def test_std_mean_is_the_mean_of_the_sample_standard_deviations() -> None:
    """DG-TSK article, Eq. (23): one spread, the mean of the per-feature standard deviations.

    The worked example of the article (three features with standard deviations 0.0837,
    0.1817 and 0.1095, giving 0.125) is only reproduced with the sample standard deviation.
    """
    x, _, _ = _data(n=5, d=3)
    points = np.arange(5)

    spreads = _pfrb_spreads(x, points, "std_mean")

    expected = np.std(x, axis=0, ddof=1).mean()
    np.testing.assert_allclose(spreads, np.full(3, expected))
    assert not np.isclose(expected, np.std(x, axis=0).mean())
    assert pytest.approx(0.125, abs=5e-5) == (0.0837 + 0.1817 + 0.1095) / 3


def test_std_mean_uses_the_rule_points_only() -> None:
    x, _, _ = _data(n=40, d=4)
    points = np.array([0, 3, 7, 11, 20])

    spreads = _pfrb_spreads(x, points, "std_mean")

    assert spreads[0] == pytest.approx(np.std(x[points], axis=0, ddof=1).mean())


def test_other_spread_modes() -> None:
    x, _, _ = _data(n=20, d=3)
    points = np.arange(20)

    np.testing.assert_allclose(_pfrb_spreads(x, points, "std"), np.std(x, axis=0))
    np.testing.assert_allclose(_pfrb_spreads(x, points, 0.4), np.full(3, 0.4))
    for bad in ("variance", 0.0, -1.0):
        with pytest.raises(ValueError, match="pfrb_spread must be"):
            _pfrb_spreads(x, points, bad)


def test_dg_tsk_starts_from_one_common_spread() -> None:
    x, _, _ = _data()

    sigmas = _initial_sigmas(DGTSKClassifier(), x)

    assert DGTSKClassifier().pfrb_spread == "std_mean"
    np.testing.assert_allclose(sigmas, np.std(x, axis=0, ddof=1).mean(), rtol=1e-6)


def test_dg_aletsk_starts_from_spreads_of_one() -> None:
    """DG-ALETSK article, Section IV: "The spreads are initialized to one"."""
    x, _, _ = _data()

    assert DGALETSKClassifier().pfrb_spread == 1.0
    np.testing.assert_allclose(_initial_sigmas(DGALETSKClassifier(), x), 1.0)
    # The regressor defaults to a clustered rule base; the spread applies to the point-based one.
    np.testing.assert_allclose(_initial_sigmas(DGALETSKRegressor(rule_base="pfrb"), x), 1.0)


def test_spread_options_and_sigma_scale() -> None:
    x, _, _ = _data()

    per_feature = _initial_sigmas(DGTSKClassifier(pfrb_spread="std"), x)
    fixed = _initial_sigmas(DGTSKClassifier(pfrb_spread=0.3, sigma_scale=2.0), x)

    np.testing.assert_allclose(per_feature[:, 0], np.std(x, axis=0), rtol=1e-6)
    np.testing.assert_allclose(fixed, 0.6)


def test_the_cache_does_not_serve_one_family_the_sets_of_another() -> None:
    """Families build different sets from the same data and arguments."""
    x, _, _ = _data()
    highfis.clear_mf_cache()

    tsk_sets, _, _ = DGTSKClassifier(random_state=0)._build_input_mfs(x)
    ale_sets, _, _ = DGALETSKClassifier(random_state=0, pfrb_max_rules=300)._build_input_mfs(x)
    changed, _, _ = DGTSKClassifier(random_state=0, pfrb_spread=0.9)._build_input_mfs(x)

    first = lambda sets: float(next(iter(sets.values()))[0].sigma.detach())  # noqa: E731
    assert first(ale_sets) == pytest.approx(1.0)
    assert first(changed) == pytest.approx(0.9)
    assert first(tsk_sets) not in (pytest.approx(1.0), pytest.approx(0.9))


# --- which samples become rules ----------------------------------------------------------


def test_rule_points_are_drawn_class_by_class() -> None:
    """Both articles: with more samples than the cap, "the stratified sampling strategy"."""
    strata = np.repeat([0, 1, 2], [300, 150, 50])

    indices = _select_pfrb_indices(len(strata), 100, random_state=0, strata=strata)

    assert np.bincount(strata[indices]).tolist() == [60, 30, 10]
    assert len(np.unique(indices)) == len(indices)
    assert np.array_equal(indices, np.sort(indices))


def test_every_class_keeps_at_least_one_rule_point() -> None:
    strata = np.repeat([0, 1], [997, 3])

    indices = _select_pfrb_indices(len(strata), 100, random_state=0, strata=strata)

    assert len(indices) == 100
    assert (strata[indices] == 1).sum() == 1


@pytest.mark.parametrize("counts", [[3, 3], [5, 4, 4], [50, 30, 20, 7], [100, 1, 1, 1]])
def test_stratified_rule_points_never_exceed_the_cap(counts: list[int]) -> None:
    strata = np.repeat(np.arange(len(counts)), counts)
    cap = max(len(counts), len(strata) // 2)

    indices = _select_pfrb_indices(len(strata), cap, random_state=0, strata=strata)

    assert len(indices) == cap
    assert set(strata[indices]) == set(range(len(counts)))


def test_rule_points_without_classes_or_below_the_cap() -> None:
    uniform = _select_pfrb_indices(500, 100, random_state=0)
    everything = _select_pfrb_indices(80, 100, random_state=0, strata=np.zeros(80, dtype=int))

    assert len(uniform) == 100
    np.testing.assert_array_equal(everything, np.arange(80))


def test_classifier_rules_and_their_labels_come_from_the_same_stratified_points() -> None:
    """The label that initializes a rule must be the label of the sample that is its centre."""
    rng = np.random.default_rng(0)
    x = rng.random((200, 4))
    y = np.repeat([0, 1], [150, 50])
    x[y == 1] += 2.0  # class 1 lives above 2, so a centre tells the class of its sample
    est = DGTSKClassifier(pfrb_max_rules=40, dg_epochs=0, finetune_epochs=0, random_state=0)

    est._pfrb_strata = y
    input_mfs, _, _ = est._build_input_mfs(x)
    del est._pfrb_strata
    centres = np.array([float(mf.mean.detach()) for mf in next(iter(input_mfs.values()))])
    labels = est._pfrb_aligned_labels(torch.as_tensor(x), torch.as_tensor(y)).numpy()

    assert len(centres) == 40
    assert (centres > 2.0).sum() == 10
    np.testing.assert_array_equal(labels, (centres > 2.0).astype(int))


def test_fit_draws_stratified_rule_points_and_cleans_up() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((120, 4))
    y = np.repeat([0, 1, 2], [60, 40, 20])
    rng.shuffle(y)

    est = DGTSKClassifier(pfrb_max_rules=30, dg_epochs=0, finetune_epochs=0, random_state=0, structural_pruning=False)
    est.fit(x, y)

    assert not hasattr(est, "_pfrb_strata")


# --- gate initialization -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "value"),
    [(DGTSKClassifier, 0.1), (DGTSKRegressor, 0.1), (DGALETSKClassifier, 0.01), (DGALETSKRegressor, 0.01)],
)
def test_every_gate_starts_at_the_constant_of_its_article(cls: Any, value: float) -> None:
    x, _, _ = _data()
    est = cls(dg_epochs=0, finetune_epochs=0, random_state=0)
    input_mfs, _, rule_base = est._build_input_mfs(x)
    model: Any = (
        est._build_model(input_mfs, 2, rule_base)
        if cls.__name__.endswith("Classifier")
        else est._build_regressor_model(input_mfs, rule_base)
    )

    assert torch.equal(model.rule_layer.lambda_gates, torch.full_like(model.rule_layer.lambda_gates, value))
    assert torch.equal(model.consequent_layer.theta_gates, torch.full_like(model.consequent_layer.theta_gates, value))


def test_fsre_adatsk_gates_start_at_the_constant_of_its_article() -> None:
    x, _, _ = _data()
    est = FSREADATSKClassifier(random_state=0)
    input_mfs, _, rule_base = est._build_input_mfs(x)

    model: Any = est._build_model(input_mfs, 2, rule_base)

    assert torch.equal(model.consequent_layer.lambda_gates, torch.full_like(model.consequent_layer.lambda_gates, 0.01))
    assert torch.equal(model.consequent_layer.theta_gates, torch.full_like(model.consequent_layer.theta_gates, 0.01))


# --- the gates leave the model after pruning ---------------------------------------------


@pytest.mark.parametrize("cls", DG_ESTIMATORS)
def test_fitted_model_runs_without_gates(cls: Any) -> None:
    """After pruning, fine-tuning and prediction run on a plain TSK system."""
    est, x = _fit(cls)
    model = est.model_
    before = est.predict(x)

    assert not bool(model.rule_layer.gates_enabled)
    assert model.consequent_layer.mode == "finetune"
    with torch.no_grad():
        model.rule_layer.lambda_gates.fill_(0.37)
        model.consequent_layer.theta_gates.fill_(0.37)
    np.testing.assert_array_equal(est.predict(x), before)


@pytest.mark.parametrize("cls", DG_ESTIMATORS)
def test_gates_stay_removed_after_save_and_load(cls: Any, tmp_path: Path) -> None:
    est, x = _fit(cls)
    path = str(tmp_path / "model.pt")

    est.save(path)
    loaded = cls.load(path)

    assert not bool(loaded.model_.rule_layer.gates_enabled)
    assert loaded.model_.consequent_layer.mode == "finetune"
    np.testing.assert_array_equal(loaded.predict(x), est.predict(x))


@pytest.mark.parametrize("cls", [DGTSKClassifier, DGALETSKClassifier])
def test_checkpoints_without_the_switch_load_with_their_gates_on(cls: Any) -> None:
    """A model saved before the switch existed used its gates in the forward pass."""
    est, _ = _fit(cls)
    state = {k: v for k, v in est.model_.state_dict().items() if not k.endswith("gates_enabled")}
    est.model_.rule_layer.gates_enabled.fill_(False)

    est.model_.load_state_dict(state)

    assert bool(est.model_.rule_layer.gates_enabled)


@pytest.mark.parametrize("cls", DG_ESTIMATORS)
def test_gate_values_remain_available_for_inspection(cls: Any) -> None:
    est, _ = _fit(cls)

    assert est.get_feature_gates().shape == (est.model_.n_inputs,)
    assert est.get_rule_gates().shape == (est.model_.n_rules,)


# --- defaults and end to end -------------------------------------------------------------


def test_dg_tsk_defaults_follow_the_article() -> None:
    for cls in (DGTSKClassifier, DGTSKRegressor):
        est = cls()
        assert (est.dg_epochs, est.finetune_epochs, est.learning_rate) == (300, 300, 0.2)
        assert est.batch_size == "auto"


def test_dg_tsk_on_iris_end_to_end() -> None:
    """DG-TSK article, Table 3: 96.8% on Iris with about two features.

    Before the fixes the package reached 66% here. The bounds are loose on purpose: they
    catch a return to that behaviour, not a change of a point or two.
    """
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
    scaler = MinMaxScaler().fit(x_train)

    clf = DGTSKClassifier(random_state=0).fit(scaler.transform(x_train), y_train)

    assert clf.score(scaler.transform(x_test), y_test) >= 0.9
    assert 1 <= len(clf.selected_features_) <= 3
