"""Sanity of the regressors on a problem with a known answer.

A first-order TSK system contains the linear model, so on an exactly linear target a
regressor far below ridge regression is either defective or not trained. Several families
were: their regressors were not built like their classifiers.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import highfis
from highfis.estimators._base import _start_from_target_mean
from highfis.memberships import ADATSKGaussianMF


def _linear_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((300, 8))
    y = x @ np.array([3.0, -2.0, 1.5, 0.0, 0.0, 1.0, -1.0, 0.5]) + 0.05 * rng.standard_normal(300)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    xs, ys = MinMaxScaler().fit(x_train), MinMaxScaler().fit(y_train.reshape(-1, 1))
    scale = lambda v: ys.transform(v.reshape(-1, 1)).ravel()  # noqa: E731
    return xs.transform(x_train), scale(y_train), xs.transform(x_test), scale(y_test)


def _built(name: str, x: np.ndarray) -> Any:
    est = getattr(highfis, name)(random_state=0)
    input_mfs, _, rule_base = est._build_input_mfs(x)
    build = est._build_regressor_model if name.endswith("Regressor") else est._build_model
    return build(input_mfs, rule_base) if name.endswith("Regressor") else build(input_mfs, 2, rule_base)


# Ridge regression reaches 0.998 here. Before the fixes these four gave -2.35, 0.26, 0.36 and 0.80.
@pytest.mark.parametrize(
    ("name", "at_least"),
    [("ADATSKRegressor", 0.6), ("ADPTSKRegressor", 0.7), ("AYATSKRegressor", 0.8), ("FSREADATSKRegressor", 0.9)],
)
def test_regressor_fits_a_linear_target(name: str, at_least: float) -> None:
    x_train, y_train, x_test, y_test = _linear_problem()

    reg = getattr(highfis, name)(random_state=0).fit(x_train, y_train)

    assert reg.score(x_test, y_test) >= at_least


def test_adatsk_regressor_is_built_like_its_classifier() -> None:
    """Zero consequents and the ADATSK Gaussian sets, as in the classifier."""
    x, _, _, _ = _linear_problem()

    model = _built("ADATSKRegressor", x)

    assert not torch.any(model.consequent_layer.weight != 0)
    assert not torch.any(model.consequent_layer.bias != 0)
    sets = [mf for mfs in model.membership_layer.input_mfs.values() for mf in mfs]
    assert all(isinstance(mf, ADATSKGaussianMF) for mf in sets)


@pytest.mark.parametrize("name", ["FSREADATSKClassifier", "FSREADATSKRegressor"])
def test_fsre_adatsk_consequents_start_at_zero(name: str) -> None:
    """Article, Section IV: "All the consequent parameters are initialized to zero"."""
    x, _, _, _ = _linear_problem()
    model = _built(name, x)
    assert not torch.any(model.consequent_layer.weight != 0)

    model.expand_to_en_frb()

    assert not torch.any(model.consequent_layer.weight != 0)
    assert not torch.any(model.consequent_layer.bias != 0)


def test_zero_initialized_regressor_starts_from_the_target_mean() -> None:
    x, y, _, _ = _linear_problem()
    model = _built("ADPTSKRegressor", x)

    _start_from_target_mean(model, torch.as_tensor(y))

    assert torch.allclose(model.consequent_layer.bias, torch.full_like(model.consequent_layer.bias, float(y.mean())))
    with torch.no_grad():
        prediction = model(torch.as_tensor(x, dtype=torch.float32)).squeeze(1)
    assert torch.allclose(prediction, torch.full_like(prediction, float(y.mean())), atol=1e-5)


def test_other_initializations_are_left_alone() -> None:
    """A family that does not start from zero keeps its own initialization."""
    x, y, _, _ = _linear_problem()
    model = _built("HTSKRegressor", x)
    weight, bias = model.consequent_layer.weight.clone(), model.consequent_layer.bias.clone()
    assert torch.any(weight != 0)

    _start_from_target_mean(model, torch.as_tensor(y))

    assert torch.equal(model.consequent_layer.weight, weight)
    assert torch.equal(model.consequent_layer.bias, bias)


def test_diverged_training_warns() -> None:
    """Plain gradient descent on an unscaled target ends in NaN; the user must be told."""
    rng = np.random.default_rng(0)
    x = rng.random((80, 4))
    y = 1000.0 + 500.0 * x[:, 0]
    reg = highfis.FSREADATSKRegressor(random_state=0, fs_epochs=30, re_epochs=30, finetune_epochs=30)

    with pytest.warns(RuntimeWarning, match="Training diverged"):
        reg.fit(x, y)


def test_normal_training_does_not_warn() -> None:
    x_train, y_train, _, _ = _linear_problem()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        highfis.HTSKRegressor(random_state=0, epochs=5).fit(x_train, y_train)
