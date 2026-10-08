"""HDFIS follows its source article (Xue et al., 2023) and its authors' code."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import highfis
from highfis.defuzzifiers import SoftmaxLogDefuzzifier, SumBasedDefuzzifier

HDFIS = ["HDFISProdClassifier", "HDFISMinClassifier", "HDFISProdRegressor", "HDFISMinRegressor"]


def _wide(n_features: int = 3000, n_samples: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((n_samples, n_features)).astype(np.float32)
    x[0], x[1] = 0.0, 1.0  # so that every feature spans [0, 1]
    return x, (x[:, :50].mean(axis=1) > 0.5).astype(int)


def _fit(name: str, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> Any:
    target = y if name.endswith("Classifier") else y.astype(np.float32)
    return getattr(highfis, name)(random_state=0, **kwargs).fit(x, target)


@pytest.mark.parametrize("name", HDFIS)
def test_defaults_are_the_partition_of_the_article(name: str) -> None:
    params = getattr(highfis, name)().get_params()
    assert (params["mf_init"], params["n_mfs"]) == ("grid", 3)


@pytest.mark.parametrize("name", HDFIS)
def test_centres_spreads_and_rules_of_the_article(name: str) -> None:
    x, y = _wide(n_features=20)
    # A learning rate of zero keeps the parameters at their initial values.
    model = _fit(name, x, y, epochs=1, learning_rate=0.0)
    for sets in model.get_mf_params().values():
        assert [mf["mean"] for mf in sets] == pytest.approx([0.0, 0.5, 1.0], abs=1e-6)
        assert [mf["sigma"] for mf in sets] == pytest.approx([1.0, 1.0, 1.0], abs=1e-5)
    assert model.model_.n_rules == 3
    assert model.rule_base_ == "coco"


@pytest.mark.parametrize("name", ["HDFISProdClassifier", "HDFISMinClassifier"])
def test_consequents_start_at_zero(name: str) -> None:
    x, y = _wide(n_features=20)
    model = _fit(name, x, y, epochs=1, learning_rate=0.0)
    assert float(np.abs(model.get_consequent_weights()).max()) == 0.0
    assert float(np.abs(model.get_consequent_bias()).max()) == 0.0


def test_the_product_does_not_underflow_in_single_precision() -> None:
    """The bound of the article holds for doubles; the logarithmic domain makes it hold for singles."""
    x, y = _wide()
    with warnings.catch_warnings():
        warnings.simplefilter("error", highfis.DegenerateFiringWarning)
        model = _fit("HDFISProdClassifier", x, y, epochs=1, learning_rate=0.0)
    weights = model.rule_activation(x)

    # Normalized product of the membership degrees of the article, in double precision.
    dimension = x.shape[1]
    scale = dimension ** (1.0 - np.log(745.0) / np.log(dimension))
    centres = np.array([0.0, 0.5, 1.0])
    exponent = -((x[:, None, :].astype(np.float64) - centres[None, :, None]) ** 2).sum(axis=2) / (scale + 1.0)
    expected = np.exp(exponent - exponent.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(weights, expected, atol=2e-4)
    assert model.firing_diagnostics(x)["uniform_fraction"] == 0.0
    # The plain product is below the smallest single-precision number for these samples.
    assert exponent.min() < np.log(np.finfo(np.float32).tiny)


def test_product_and_log_domain_agree_in_low_dimension() -> None:
    x, y = _wide(n_features=6)
    model = _fit("HDFISProdClassifier", x, y, epochs=3)
    with torch.no_grad():
        degrees = torch.stack(list(model.model_.membership_layer(torch.tensor(x)).values()), dim=1)  # (N, D, R)
    product = degrees.prod(dim=1)
    expected = (product / product.sum(dim=1, keepdim=True)).numpy()
    np.testing.assert_allclose(model.rule_activation(x), expected, atol=1e-6)


def test_sum_based_normalization_keeps_the_ratios_of_small_strengths() -> None:
    strengths = torch.tensor([[1e-20, 3e-20], [2e-30, 2e-30], [0.0, 0.0]])
    weights = SumBasedDefuzzifier()(strengths)
    assert weights[0].tolist() == pytest.approx([0.25, 0.75])
    assert weights[1].tolist() == pytest.approx([0.5, 0.5])
    assert weights[2].tolist() == pytest.approx([0.5, 0.5])  # nothing fires: uniform, and finite


def test_softmax_log_scale_is_a_power() -> None:
    strengths = torch.tensor([[0.5, 0.25]])
    weights = SoftmaxLogDefuzzifier(scale=3.0)(strengths)
    expected = torch.tensor([0.5**3, 0.25**3])
    assert weights[0].tolist() == pytest.approx((expected / expected.sum()).tolist())
    assert SoftmaxLogDefuzzifier()(strengths)[0].tolist() == pytest.approx([2 / 3, 1 / 3])


@pytest.mark.parametrize("name", ["HDFISProdClassifier", "HDFISMinClassifier"])
def test_clustering_is_still_available(name: str) -> None:
    x, y = _wide(n_features=20)
    model = _fit(name, x, y, epochs=1, mf_init="kmeans", n_mfs=4)
    assert model.model_.n_rules == 4
    centres = [mf["mean"] for mf in model.get_mf_params()["x1"]]
    assert centres != pytest.approx([0.0, 1 / 3, 2 / 3, 1.0], abs=1e-3)


@pytest.mark.parametrize("name", HDFIS)
def test_predictions_survive_save_and_load(name: str, tmp_path: Path) -> None:
    x, y = _wide(n_features=30)
    model = _fit(name, x, y, epochs=3)
    path = str(tmp_path / "model.pt")
    model.save(path)
    loaded = getattr(highfis, name).load(path)
    np.testing.assert_allclose(loaded.predict(x), model.predict(x), rtol=1e-6, atol=1e-6)
