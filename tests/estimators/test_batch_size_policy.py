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


# --- weight_decay contract -------------------------------------------------------------
# weight_decay is documented as "L2 weight decay for consequent parameters" on every
# estimator, but several families used to drop it in _get_optimizer_config, making it a
# silent no-op. These guards pin that it reaches the consequent group (and only that
# group) for *every* family.


def _model_of(name: str):  # type: ignore[no-untyped-def]
    """Fit a tiny model of the named estimator and return its underlying module."""
    rng = np.random.default_rng(0)
    x = rng.random((40, 4)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(np.int64)
    extra = {}
    n = name
    if "ADPTSK" in n or "ADMTSK" in n or "AYATSK" in n:
        extra["k"] = 1.5
    if "DombiTSK" in n:
        extra["lambda_"] = 1.5
    est = getattr(highfis, name)(n_mfs=2, random_state=0, **extra)
    return est.fit(x, y).model_


@pytest.mark.parametrize("name", [n for n in ESTIMATORS if n.endswith("Classifier")])
def test_weight_decay_reaches_the_consequent_group_for_every_family(name: str) -> None:
    """``weight_decay`` must land on the consequent parameters, whatever the optimizer."""
    from highfis.optim._utils import _get_optimizer_config

    model = _model_of(name)
    _opt_cls, groups = _get_optimizer_config(model, learning_rate=0.01, weight_decay=0.5)

    # The consequent group is the consequent layer plus its optional BatchNorm.
    cons_ids = {id(p) for p in model.consequent_layer.parameters()}
    if model.consequent_bn is not None:
        cons_ids |= {id(p) for p in model.consequent_bn.parameters()}
    decayed = [wd for g in groups if (wd := g.get("weight_decay", 0.0)) > 0.0]
    assert decayed, f"{name}: weight_decay dropped for every group (silent no-op)"
    for g in groups:
        group_ids = {id(p) for p in g["params"]}
        if g.get("weight_decay", 0.0) > 0.0:
            # a decayed group must be the consequent one, not antecedent/rule
            assert group_ids <= cons_ids, f"{name}: weight_decay applied outside the consequent group"


def test_weight_decay_is_a_no_op_at_zero_but_real_when_set() -> None:
    """End-to-end: a previously-broken family (ADPTSK) shrinks the consequent under decay."""
    import torch

    def cons_norm(wd: float) -> float:
        model = highfis.ADPTSKClassifier
        rng = np.random.default_rng(0)
        x = rng.random((200, 8)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] > 1).astype(np.int64)
        m = model(n_mfs=2, epochs=40, learning_rate=0.05, k=1.5, weight_decay=wd, random_state=0).fit(x, y)
        with torch.no_grad():
            return float(torch.cat([p.flatten() for p in m.model_.consequent_layer.parameters()]).norm())

    assert cons_norm(1.0) < 0.5 * cons_norm(0.0)


# --- float64 end-to-end ----------------------------------------------------------------
# Model parameters follow torch's default dtype, but the input/target/prediction paths
# used to hard-code float32, crashing with a Float/Double mismatch in the consequent
# layer when a user set torch.set_default_dtype(torch.float64) before fitting.


@pytest.mark.parametrize(
    "name,extra",
    [
        ("HTSKClassifier", {}),
        ("ADPTSKClassifier", {"k": 1.5}),
        ("ADATSKClassifier", {}),
        ("MHTSKClassifier", {"n_heads": 3}),
        ("FSREADATSKClassifier", {"fs_epochs": 4, "re_epochs": 4, "finetune_epochs": 4}),
        ("DGALETSKClassifier", {"dg_epochs": 4, "finetune_epochs": 4}),
    ],
)
def test_fit_and_predict_in_float64(name: str, extra: dict) -> None:  # type: ignore[type-arg]
    """Training and inference run end to end under a float64 default dtype."""
    import torch

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        rng = np.random.default_rng(0)
        x = rng.random((60, 5))  # float64 numpy
        y = (x[:, 0] > 0.5).astype(np.int64)
        kw = {"n_mfs": 2, "random_state": 0, "verbose": False, **extra}
        if "epochs" not in kw and not any(k.endswith("epochs") for k in extra):
            kw["epochs"] = 6

        clf = getattr(highfis, name)(**kw).fit(x, y)

        param = next(clf.model_.parameters())
        assert param.dtype == torch.float64, f"{name} did not train in float64"
        preds = clf.predict(x)
        assert preds.shape == (60,)
        assert 0.0 <= clf.score(x, y) <= 1.0
    finally:
        torch.set_default_dtype(prev)


def test_default_float32_path_is_unchanged() -> None:
    """The normal single-precision path still produces float32 models."""
    import torch

    assert torch.get_default_dtype() == torch.float32  # test isolation sanity
    rng = np.random.default_rng(0)
    x = rng.random((40, 4)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(np.int64)
    clf = highfis.HTSKClassifier(n_mfs=2, epochs=3, random_state=0).fit(x, y)
    assert next(clf.model_.parameters()).dtype == torch.float32


def test_model_dtype_falls_back_to_default_before_fit() -> None:
    """Before fitting there is no model, so the input dtype follows the global default."""
    import torch

    est = highfis.HTSKClassifier(n_mfs=2)
    assert est._model_dtype() == torch.get_default_dtype()

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        assert est._model_dtype() == torch.float64
    finally:
        torch.set_default_dtype(prev)
