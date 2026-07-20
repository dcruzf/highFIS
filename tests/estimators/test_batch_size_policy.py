"""Cross-estimator guards for the ``batch_size`` contract and the paper-derived defaults.

``batch_size`` is three-valued: an explicit ``int`` (or ``None`` for full batch) always
wins, while ``"auto"`` — the default — defers to the source paper's policy for that
family, resolved from the training-set size at ``fit`` time.

The policies differ per paper, so a single library-wide number would contradict most of
them. These tests pin each family's policy and the precedence rules, so neither can drift.
"""

from __future__ import annotations

import numpy as np
import pytest

import highfis

ESTIMATORS: list[str] = sorted(n for n in highfis.__all__ if n.endswith(("Classifier", "Regressor")))

# N values that matter here: Colon's training split, Iris, and a set large enough to pass
# the 500-sample thresholds in the ADPTSK/AYATSK protocols.
_N_COLON, _N_IRIS, _N_LARGE = 43, 150, 1000

# family prefix -> expected (N=43, N=150, N=1000) from the paper cited in the source module.
_EXPECTED: dict[str, tuple[int | None, int | None, int | None]] = {
    # HTSK_2021: 512, clamped to min(N, 60) when it exceeds the training set.
    "HTSK": (43, 60, 512),
    "LogTSK": (43, 60, 512),
    "TSK": (43, 60, 512),
    # HDFIS_2023: fixed 64.
    "HDFISProd": (64, 64, 64),
    "HDFISMin": (64, 64, 64),
    # 10% of N (DG-ALETSK_2023; ADMTSK_2025 for the Dombi family).
    "DGALETSK": (4, 15, 100),
    "DombiTSK": (4, 15, 100),
    "ADMTSK": (4, 15, 100),
    # Papers prescribing full-batch gradient descent.
    "DGTSK": (None, None, None),
    "FSREADATSK": (None, None, None),
    "ADATSK": (None, None, None),
    # Full batch below 500 samples, then a fraction of N.
    "ADPTSK": (None, None, 200),
    "AYATSK": (None, None, 100),
    # MHTSK_2025 specifies no batch size; highFIS defaults to 64.
    "MHTSK": (64, 64, 64),
}


def _family(name: str) -> str:
    return name.replace("Classifier", "").replace("Regressor", "")


def test_every_estimator_has_a_declared_policy() -> None:
    """Guard the table itself: a new family must be given an explicit expectation."""
    assert len(ESTIMATORS) >= 28
    missing = sorted({_family(n) for n in ESTIMATORS} - set(_EXPECTED))
    assert not missing, f"families without a declared batch-size policy: {missing}"


@pytest.mark.parametrize("name", ESTIMATORS)
def test_paper_batch_size_matches_the_source_paper(name: str) -> None:
    """``"auto"`` must resolve to the batch size the family's paper prescribes."""
    est = getattr(highfis, name)()
    expected = _EXPECTED[_family(name)]
    got = tuple(est._paper_batch_size(n) for n in (_N_COLON, _N_IRIS, _N_LARGE))
    assert got == expected, f"{name} batch policy drifted: {got} != {expected}"


@pytest.mark.parametrize("name", ESTIMATORS)
def test_explicit_batch_size_always_wins(name: str) -> None:
    """An explicit ``int`` overrides the paper policy, whatever the dataset size."""
    est = getattr(highfis, name)(batch_size=7)
    for n in (_N_COLON, _N_IRIS, _N_LARGE):
        assert est._resolve_batch_size(n) == 7


@pytest.mark.parametrize("name", ESTIMATORS)
def test_none_means_full_batch_not_the_policy(name: str) -> None:
    """``None`` keeps its documented meaning (full batch) and does *not* trigger the policy.

    This is why the ``"auto"`` sentinel exists: families with a numeric policy (HDFIS=64)
    would otherwise have no way to express "use the whole dataset".
    """
    est = getattr(highfis, name)(batch_size=None)
    for n in (_N_COLON, _N_IRIS, _N_LARGE):
        assert est._resolve_batch_size(n) is None


@pytest.mark.parametrize("name", ESTIMATORS)
def test_auto_is_the_default_and_defers_to_the_policy(name: str) -> None:
    est = getattr(highfis, name)()
    assert est.batch_size == "auto"
    for n in (_N_COLON, _N_IRIS, _N_LARGE):
        assert est._resolve_batch_size(n) == est._paper_batch_size(n)


def test_fit_exposes_the_resolved_batch_size_without_mutating_the_parameter() -> None:
    """``batch_size_`` reports what training used; ``batch_size`` stays as constructed.

    Before this contract the dataset-dependent policies were invisible, and they were
    implemented by assigning to ``self.batch_size`` during ``fit``.
    """
    rng = np.random.default_rng(0)
    x = rng.random((60, 4)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(np.int64)

    clf = highfis.HDFISProdClassifier(n_mfs=2, epochs=1)
    clf.fit(x, y)
    assert clf.batch_size == "auto"  # parameter untouched by fit
    assert clf.batch_size_ == 64  # HDFIS paper policy

    explicit = highfis.HDFISProdClassifier(n_mfs=2, epochs=1, batch_size=8)
    explicit.fit(x, y)
    assert explicit.batch_size_ == 8


def test_htsk_regressor_honours_full_batch() -> None:
    """Regression: ``HTSKRegressor`` used to silently coerce ``None`` to 512."""
    rng = np.random.default_rng(0)
    x = rng.random((40, 3)).astype(np.float32)
    y = x[:, 0].astype(np.float32)

    reg = highfis.HTSKRegressor(n_mfs=2, epochs=1, batch_size=None)
    reg.fit(x, y)
    assert reg.batch_size_ is None
