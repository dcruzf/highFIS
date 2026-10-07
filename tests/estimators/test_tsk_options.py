"""Selectable building blocks of the generic TSK estimators.

``TSKClassifier`` and ``TSKRegressor`` let the user choose the membership-function shape,
the T-norm and the defuzzifier. The named families keep their fixed combination, so these
tests also pin the two that the generic TSK can express: HTSK and LogTSK.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.base import clone

from highfis import HTSKClassifier, HTSKRegressor, LogTSKClassifier, TSKClassifier, TSKRegressor
from highfis.estimators._htsk import _MF_BUILDERS, _convert_input_mfs
from highfis.memberships import BellMF, GaussianMF, GaussianPiMF, TrapezoidalMF, TriangularMF

MF_TYPES = {
    "gaussian": GaussianMF,
    "gaussian_pi": GaussianPiMF,
    "bell": BellMF,
    "triangular": TriangularMF,
    "trapezoidal": TrapezoidalMF,
}
T_NORMS = ["prod", "min", "gmean", "dombi", "yager", "yager_simple", "ale_softmin_yager"]
DEFUZZIFIERS = ["sum", "softmax_log", "log_sum", "inv_log"]


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((120, 4))
    return x, (x[:, 0] + x[:, 1] > 1.0).astype(int), 2.0 * x[:, 0] + x[:, 1]


def _mf_types(est: Any) -> set[type]:
    return {type(mf) for mfs in est.model_.membership_layer.input_mfs.values() for mf in mfs}


def test_builders_cover_the_documented_shapes() -> None:
    assert set(_MF_BUILDERS) == set(MF_TYPES)


@pytest.mark.parametrize("name", ["bell", "triangular", "trapezoidal"])
def test_shapes_share_centre_and_half_maximum_width_with_the_gaussian(name: str) -> None:
    """Changing ``mf`` changes the shape only: same centre, same width at half maximum."""
    centre, sigma = 0.3, 0.2
    half_width = math.sqrt(2.0 * math.log(2.0)) * sigma
    mf = _MF_BUILDERS[name](centre, sigma)
    x = torch.tensor([centre, centre - half_width, centre + half_width])

    assert mf(x).tolist() == pytest.approx([1.0, 0.5, 0.5], abs=1e-6)


def test_gaussian_pi_keeps_mean_and_sigma() -> None:
    mf = _MF_BUILDERS["gaussian_pi"](0.3, 0.2)

    assert isinstance(mf, GaussianPiMF)
    assert float(mf.mean.detach()) == pytest.approx(0.3)
    assert float(mf.sigma.detach()) == pytest.approx(0.2)


def test_conversion_leaves_non_gaussian_sets_untouched() -> None:
    """Reloading builds the model from sets that already have the chosen type."""
    already = {"x1": [TriangularMF(left=0.0, center=0.5, right=1.0)]}

    assert _convert_input_mfs(already, "bell")["x1"][0] is already["x1"][0]


@pytest.mark.parametrize("mf", list(MF_TYPES))
def test_classifier_and_regressor_use_the_chosen_shape(mf: str) -> None:
    x, y, y_reg = _data()

    clf = TSKClassifier(n_mfs=3, epochs=3, random_state=0, mf=mf).fit(x, y)
    reg = TSKRegressor(n_mfs=3, epochs=3, random_state=0, mf=mf).fit(x, y_reg)

    assert _mf_types(clf) == {MF_TYPES[mf]}
    assert _mf_types(reg) == {MF_TYPES[mf]}
    assert np.isfinite(clf.predict_proba(x)).all()
    assert np.isfinite(reg.predict(x)).all()


@pytest.mark.parametrize("t_norm", T_NORMS)
@pytest.mark.parametrize("defuzzifier", DEFUZZIFIERS)
def test_every_t_norm_and_defuzzifier_trains(t_norm: str, defuzzifier: str) -> None:
    x, y, y_reg = _data()

    clf = TSKClassifier(n_mfs=3, epochs=3, random_state=0, t_norm=t_norm, defuzzifier=defuzzifier).fit(x, y)
    reg = TSKRegressor(n_mfs=3, epochs=3, random_state=0, t_norm=t_norm, defuzzifier=defuzzifier).fit(x, y_reg)

    assert np.isfinite(clf.predict_proba(x)).all()
    assert np.isfinite(reg.predict(x)).all()
    assert np.isfinite(clf.history_["train_loss"]).all()


@pytest.mark.parametrize("mf", ["triangular", "trapezoidal"])
def test_bounded_support_stays_finite_far_from_the_data(mf: str) -> None:
    """Outside the support of every rule the membership is zero; the output must stay finite."""
    x, y, y_reg = _data()
    far = np.vstack([x[:5] + 10.0, x[:5] - 10.0])

    clf = TSKClassifier(n_mfs=3, epochs=3, random_state=0, mf=mf).fit(x, y)
    reg = TSKRegressor(n_mfs=3, epochs=3, random_state=0, mf=mf).fit(x, y_reg)

    proba = clf.predict_proba(far)
    assert np.isfinite(proba).all()
    assert proba.sum(axis=1) == pytest.approx(1.0)
    assert np.isfinite(reg.predict(far)).all()


def test_defaults_are_the_classical_tsk() -> None:
    """Gaussian sets, product T-norm and sum normalization, spelled out or left implicit."""
    x, y, _ = _data()

    implicit = TSKClassifier(n_mfs=3, epochs=5, random_state=0).fit(x, y)
    explicit = TSKClassifier(n_mfs=3, epochs=5, random_state=0, mf="gaussian", t_norm="prod", defuzzifier="sum").fit(
        x, y
    )

    assert (implicit.mf, implicit.t_norm, implicit.defuzzifier) == ("gaussian", "prod", "sum")
    np.testing.assert_array_equal(implicit.predict_proba(x), explicit.predict_proba(x))


def test_geometric_mean_with_softmax_log_is_htsk() -> None:
    x, y, y_reg = _data()
    options: dict[str, Any] = {"t_norm": "gmean", "defuzzifier": "softmax_log"}

    tsk = TSKClassifier(n_mfs=3, epochs=5, random_state=0, **options).fit(x, y)
    htsk = HTSKClassifier(n_mfs=3, epochs=5, random_state=0).fit(x, y)
    tsk_reg = TSKRegressor(n_mfs=3, epochs=5, random_state=0, **options).fit(x, y_reg)
    htsk_reg = HTSKRegressor(n_mfs=3, epochs=5, random_state=0).fit(x, y_reg)

    np.testing.assert_array_equal(tsk.predict_proba(x), htsk.predict_proba(x))
    np.testing.assert_array_equal(tsk_reg.predict(x), htsk_reg.predict(x))


def test_geometric_mean_with_inverse_log_is_logtsk() -> None:
    x, y, _ = _data()

    tsk = TSKClassifier(n_mfs=3, epochs=5, random_state=0, t_norm="gmean", defuzzifier="inv_log").fit(x, y)
    logtsk = LogTSKClassifier(n_mfs=3, epochs=5, random_state=0).fit(x, y)

    np.testing.assert_array_equal(tsk.predict_proba(x), logtsk.predict_proba(x))


@pytest.mark.parametrize(
    ("option", "match"),
    [
        ({"mf": "sigmoid"}, "mf must be one of"),
        ({"t_norm": "lukasiewicz"}, "t_norm must be"),
        ({"defuzzifier": "centroid"}, "defuzzifier must be one of"),
    ],
)
def test_unknown_option_is_rejected_at_fit(option: dict[str, Any], match: str) -> None:
    x, y, y_reg = _data()

    with pytest.raises(ValueError, match=match):
        TSKClassifier(n_mfs=3, epochs=1, **option).fit(x, y)
    with pytest.raises(ValueError, match=match):
        TSKRegressor(n_mfs=3, epochs=1, **option).fit(x, y_reg)


@pytest.mark.parametrize("mf", ["gaussian_pi", "bell", "triangular", "trapezoidal"])
def test_options_survive_clone_and_save_load(mf: str, tmp_path: Path) -> None:
    x, y, _ = _data()
    options: dict[str, Any] = {"mf": mf, "t_norm": "min", "defuzzifier": "softmax_log"}
    clf = TSKClassifier(n_mfs=3, epochs=3, random_state=0, **options).fit(x, y)

    cloned = clone(clf)
    path = tmp_path / "tsk.pt"
    clf.save(str(path))
    loaded = TSKClassifier.load(str(path))

    assert {k: getattr(cloned, k) for k in options} == options
    assert {k: getattr(loaded, k) for k in options} == options
    assert _mf_types(loaded) == {MF_TYPES[mf]}
    np.testing.assert_array_equal(loaded.predict_proba(x), clf.predict_proba(x))


@pytest.mark.parametrize("mf", ["bell", "triangular"])
def test_introspection_reports_the_chosen_shape(mf: str) -> None:
    x, y, _ = _data()
    clf = TSKClassifier(n_mfs=3, epochs=3, random_state=0, mf=mf).fit(x, y)

    first = next(iter(clf.get_mf_params().values()))[0]

    assert first["type"] == MF_TYPES[mf].__name__
    assert clf.rule_activation(x[:4]).shape == (4, 3)
    assert clf.feature_importance() is not None
