"""The three membership functions that the ADMTSK article compares."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

import highfis

ESTIMATORS = ["ADMTSKClassifier", "ADMTSKRegressor"]
KINDS = {"composite": "GaussianPiMF", "extended": "ExtendedGaussianMF", "traditional": "GaussianMF"}


def _fit(name: str, **params: Any) -> tuple[Any, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((60, 40)).astype(np.float32)
    y: Any = (x[:, 0] > 0.5).astype(int) if name.endswith("Classifier") else x[:, 0]
    return getattr(highfis, name)(epochs=3, random_state=0, **params).fit(x, y), x


def _t_norm_parameter(est: Any) -> float:
    return float(est.model_.rule_layer._resolved_t_norm.dombi.lambda_)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("membership", list(KINDS))
def test_each_membership_builds_its_fuzzy_sets(name: str, membership: str) -> None:
    est, _ = _fit(name, membership=membership)

    assert {type(mf).__name__ for mfs in est.model_.input_mfs.values() for mf in mfs} == {KINDS[membership]}
    # The three functions have a vectorized kernel, so none falls back to the per-set loop.
    assert est.model_.membership_layer._fast_kernel is not None


def test_the_composite_set_is_the_default() -> None:
    assert highfis.ADMTSKClassifier().get_params()["membership"] == "composite"


@pytest.mark.parametrize("bound", [0.1, 0.3, 0.5])
def test_the_extended_set_takes_the_lower_bound_of_the_estimator(bound: float) -> None:
    est, x = _fit("ADMTSKClassifier", membership="extended", lower_bound=bound)
    degrees = est.model_.membership_layer(torch.as_tensor(x))

    assert all(mf.lower_bound == bound for mfs in est.model_.input_mfs.values() for mf in mfs)
    assert min(float(values.detach().min()) for values in degrees.values()) >= bound


def test_the_t_norm_parameter_follows_the_lower_bound_of_the_set() -> None:
    """lambda = ln D / (ln(K - eps) - ln(1 - eps)), with eps the lower bound of the set (K = 10, D = 40)."""

    def expected(bound: float) -> float:
        return math.log(40) / (math.log(10.0 - bound) - math.log(1.0 - bound))

    composite, _ = _fit("ADMTSKClassifier")
    extended, _ = _fit("ADMTSKClassifier", membership="extended", lower_bound=0.3)
    traditional, _ = _fit("ADMTSKClassifier", membership="traditional")

    assert _t_norm_parameter(composite) == pytest.approx(expected(1.0 / math.e))
    assert _t_norm_parameter(extended) == pytest.approx(expected(0.3))
    assert _t_norm_parameter(traditional) == pytest.approx(expected(0.0))


def test_an_unknown_membership_is_rejected() -> None:
    with pytest.raises(ValueError, match="membership must be one of"):
        _fit("ADMTSKClassifier", membership="triangular")


@pytest.mark.parametrize("membership", ["extended", "traditional"])
def test_a_model_with_another_membership_is_saved_and_loaded(membership: str, tmp_path: Any) -> None:
    est, x = _fit("ADMTSKClassifier", membership=membership, lower_bound=0.3)
    path = str(tmp_path / "model.pt")
    est.save(path)
    loaded = highfis.ADMTSKClassifier.load(path)

    assert loaded.membership == membership
    np.testing.assert_allclose(loaded.predict_proba(x), est.predict_proba(x), rtol=1e-6, atol=1e-7)
