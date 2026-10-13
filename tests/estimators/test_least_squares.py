"""Consequents by least squares: the solver of the models and the ``use_lse`` option of the estimators."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.base import clone

import highfis
from highfis.models import _base as model_base

ESTIMATORS: list[str] = sorted(n for n in highfis.__all__ if n.endswith(("Classifier", "Regressor")))
#: DG-TSK and DG-ALETSK solve their consequents inside their own training procedure.
WITH_OPTION: list[str] = [n for n in ESTIMATORS if not n.startswith("DG")]


def _data(name: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((120, 5)).astype(np.float32)
    if name.endswith("Classifier"):
        return x, (x[:, 0] + x[:, 1] > 1.0).astype(int)
    return x, (2.0 * x[:, 0] - x[:, 1] + 0.5).astype(np.float32)


def _fit(name: str, **params: Any) -> Any:
    x, y = _data(name)
    cls = getattr(highfis, name)
    epochs = {p: 2 for p in inspect.signature(cls.__init__).parameters if p.endswith("epochs")}
    return cls(random_state=0, **{**epochs, **params}).fit(x, y)


def _weights(est: Any) -> np.ndarray:
    weights = est.get_consequent_weights()
    assert weights is not None
    return np.asarray(weights)


def _model_error(est: Any, name: str) -> float:
    """Mean squared error of the model output against the targets the solver uses."""
    x, y = _data(name)
    inputs = torch.as_tensor(est._select_model_features(x))
    was_training = est.model_.training
    est.model_.eval()
    with torch.no_grad():
        output = est.model_(inputs)
    est.model_.train(was_training)
    if name.endswith("Classifier"):
        target = torch.nn.functional.one_hot(torch.as_tensor(y), output.shape[1]).to(output.dtype)
    else:
        target = torch.as_tensor(y).reshape(-1, 1)
    return float(((output - target) ** 2).mean())


@pytest.mark.parametrize("name", WITH_OPTION)
def test_every_family_has_the_option(name: str) -> None:
    params = getattr(highfis, name)().get_params()

    assert params["use_lse"] is False
    assert params["lse_ridge"] == 0.0


@pytest.mark.parametrize("name", WITH_OPTION)
def test_the_stored_consequents_are_the_solution(name: str) -> None:
    """The error reported by the solver is the error of the model with the new consequents."""
    est = _fit(name, use_lse=True)

    assert est.history_["lse"]["ridge"] == 0.0
    assert _model_error(est, name) == pytest.approx(est.history_["lse"]["train_mse"], rel=1e-3, abs=1e-6)


@pytest.mark.parametrize("name", WITH_OPTION)
def test_least_squares_does_not_fit_worse_than_two_epochs(name: str) -> None:
    assert _model_error(_fit(name, use_lse=True), name) <= _model_error(_fit(name), name) + 1e-6


@pytest.mark.parametrize("name", ["TSKRegressor", "HTSKRegressor", "HDFISMinRegressor", "ADMTSKRegressor"])
def test_a_linear_target_is_fitted_exactly(name: str) -> None:
    x, y = _data(name)
    est = _fit(name, use_lse=True)

    assert est.score(x, y) == pytest.approx(1.0, abs=1e-6)


def test_antecedents_are_left_as_they_are() -> None:
    plain, solved = _fit("TSKRegressor"), _fit("TSKRegressor", use_lse=True)

    assert plain.get_mf_params() == solved.get_mf_params()
    assert not np.allclose(_weights(plain), _weights(solved))


def test_zero_epochs_solve_the_consequents_on_the_initial_antecedents() -> None:
    x, y = _data("HDFISMinClassifier")
    est = highfis.HDFISMinClassifier(epochs=0, use_lse=True, random_state=0).fit(x, y)

    assert est.score(x, y) > 0.9
    assert "lse" in est.history_


def test_ridge_shrinks_the_consequents() -> None:
    free, penalized = _fit("TSKClassifier", use_lse=True), _fit("TSKClassifier", use_lse=True, lse_ridge=10.0)

    assert np.abs(_weights(penalized)).sum() < np.abs(_weights(free)).sum()
    assert penalized.history_["lse"]["train_mse"] > free.history_["lse"]["train_mse"]
    assert penalized.history_["lse"]["ridge"] == 10.0


def test_more_parameters_than_samples() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((20, 200)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(int)
    for ridge in (0.0, 1e-3):
        est = highfis.HDFISProdClassifier(epochs=0, use_lse=True, lse_ridge=ridge, random_state=0).fit(x, y)
        assert est.score(x, y) == 1.0
        assert np.isfinite(_weights(est)).all()


def test_ridge_agrees_with_the_normal_equations() -> None:
    rng = np.random.default_rng(1)
    design = torch.as_tensor(rng.random((30, 4)))
    target = torch.as_tensor(rng.random((30, 2)))
    expected = torch.linalg.solve(design.T @ design + 0.5 * torch.eye(4, dtype=design.dtype), design.T @ target)

    np.testing.assert_allclose(model_base._least_squares(design, target, 0.5, 1e-12), expected, rtol=1e-9)
    # With fewer samples than columns the same solution comes from the smaller system.
    wide, wide_target = design.T.contiguous(), target[:4]
    expected = torch.linalg.solve(wide.T @ wide + 0.5 * torch.eye(30, dtype=wide.dtype), wide.T @ wide_target)
    solution = model_base._least_squares(wide, wide_target, 0.5, 1e-12)
    np.testing.assert_allclose(solution, expected, rtol=1e-8, atol=1e-10)


def test_a_negative_ridge_is_rejected() -> None:
    x, y = _data("TSKRegressor")
    with pytest.raises(ValueError, match="non-negative"):
        highfis.TSKRegressor(epochs=1, use_lse=True, lse_ridge=-1.0, random_state=0).fit(x, y)


def test_active_gates_are_rejected() -> None:
    est = _fit("FSREADATSKClassifier")
    x, y = _data("FSREADATSKClassifier")
    est.model_.set_consequent_mode("re")

    with pytest.raises(ValueError, match="gates are switched off"):
        est.model_.solve_consequents(torch.as_tensor(est._select_model_features(x)), torch.as_tensor(y))


def test_a_problem_too_large_for_memory_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_base, "_MAX_LSE_ELEMENTS", 100)
    x, y = _data("TSKRegressor")

    with pytest.raises(ValueError, match="gradient descent instead"):
        highfis.TSKRegressor(epochs=1, use_lse=True, random_state=0).fit(x, y)


def test_the_option_is_cloned_and_saved(tmp_path: Any) -> None:
    x, y = _data("HTSKClassifier")
    est = clone(highfis.HTSKClassifier(epochs=2, use_lse=True, lse_ridge=0.1, random_state=0)).fit(x, y)
    path = str(tmp_path / "model.pt")
    est.save(path)
    loaded = highfis.HTSKClassifier.load(path)

    assert (loaded.use_lse, loaded.lse_ridge) == (True, 0.1)
    np.testing.assert_allclose(loaded.predict_proba(x), est.predict_proba(x), rtol=1e-6)


def test_the_solution_survives_rule_extraction_and_retraining() -> None:
    x, y = _data("MHTSKClassifier")
    est = highfis.MHTSKClassifier(epochs=2, rule_extraction=True, use_lse=True, random_state=0).fit(x, y)

    assert "lse" in est.history_
    assert _model_error(est, "MHTSKClassifier") == pytest.approx(est.history_["lse"]["train_mse"], rel=1e-3, abs=1e-6)


def test_the_summary_reports_the_solution() -> None:
    assert "least squares" not in _fit("TSKClassifier").summary()
    text = _fit("TSKClassifier", use_lse=True, lse_ridge=0.5).summary()

    assert "least squares      consequents solved at the end, ridge 0.5, training error" in text


def _leave_one_out_error(design: torch.Tensor, target: torch.Tensor, ridge: float) -> float:
    """Leave-one-out squared error of ridge regression, by refitting once per sample."""
    error = 0.0
    for i in range(design.shape[0]):
        keep = [j for j in range(design.shape[0]) if j != i]
        solution = model_base._least_squares(design[keep], target[keep], ridge, 1e-12)
        error += float((design[i : i + 1] @ solution - target[i : i + 1]).square().sum())
    return error


@pytest.mark.parametrize("shape", [(40, 300), (120, 8), (25, 25)])
def test_the_automatic_ridge_minimizes_the_leave_one_out_error(shape: tuple[int, int]) -> None:
    """The closed form picks the candidate that refitting sample by sample would pick."""
    generator = torch.Generator().manual_seed(0)
    design = torch.randn(shape, dtype=torch.float64, generator=generator)
    coefficients = torch.zeros((shape[1], 1), dtype=torch.float64)
    coefficients[:5] = 1.0
    target = design @ coefficients + 0.5 * torch.randn((shape[0], 1), dtype=torch.float64, generator=generator)

    chosen = model_base._select_ridge(design, target)

    candidates = (model_base._RIDGE_GRID * torch.linalg.svdvals(design).max() ** 2).tolist()
    errors = [_leave_one_out_error(design, target, ridge) for ridge in candidates]
    assert chosen == pytest.approx(candidates[int(np.argmin(errors))], rel=1e-9)


def test_the_automatic_ridge_generalizes_better_than_no_ridge() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 150)).astype(np.float32)
    y = (x[:, 0] - x[:, 1] + 0.5 * rng.normal(size=120)).astype(np.float32)
    train, test = slice(0, 40), slice(40, 120)
    scores = {}
    for ridge in (0.0, "auto"):
        est = highfis.HTSKRegressor(n_mfs=2, epochs=0, use_lse=True, lse_ridge=ridge, random_state=0)
        scores[ridge] = est.fit(x[train], y[train]).score(x[test], y[test])
        if ridge == "auto":
            assert est.history_["lse"]["ridge"] > 0.0
            assert est.model_.lse_ridge_ == est.history_["lse"]["ridge"]
    assert scores["auto"] > scores[0.0]


def test_the_automatic_ridge_works_for_a_classifier_and_is_cloned() -> None:
    est = _fit("HDFISProdClassifier", use_lse=True, lse_ridge="auto")
    assert est.history_["lse"]["ridge"] > 0.0
    assert clone(est).lse_ridge == "auto"
    assert "least squares" in est.summary()


def test_an_unknown_ridge_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="ridge must be a non-negative number or 'auto'"):
        _fit("TSKRegressor", use_lse=True, lse_ridge="best")
