from __future__ import annotations

import pytest
import torch

from highfis.models._common import _build_first_order_design_matrix


def test_build_first_order_design_matrix_shape_validations() -> None:
    norm_w = torch.ones((2, 3))
    x = torch.ones((2, 4))
    bad_feature_gates = torch.ones((3, 5))
    rule_gates = torch.ones(3)
    with pytest.raises(ValueError, match="feature_gates must have shape"):
        _build_first_order_design_matrix(norm_w, x, bad_feature_gates, rule_gates)
    feature_gates = torch.ones((3, 4))
    bad_rule_gates = torch.ones(2)
    with pytest.raises(ValueError, match="rule_gates must have shape"):
        _build_first_order_design_matrix(norm_w, x, feature_gates, bad_rule_gates)


def test_base_tsk_classifier_model_double_precision_fallback() -> None:
    from highfis.memberships import GaussianMF
    from highfis.models import HTSKClassifierModel

    input_mfs = {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(2)] for i in range(3)}
    model = HTSKClassifierModel(input_mfs, n_classes=3)
    x = torch.randn(8, 3, dtype=torch.float64)
    proba = model.predict_proba(x)
    pred = model.predict(x)
    assert proba.dtype == torch.float64
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)


def test_base_tsk_regressor_model_double_precision_fallback() -> None:
    from highfis.memberships import GaussianMF
    from highfis.models import HTSKRegressorModel

    input_mfs = {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(2)] for i in range(3)}
    model = HTSKRegressorModel(input_mfs)
    x = torch.randn(8, 3, dtype=torch.float64)
    pred = model.predict(x)
    assert pred.dtype == torch.float64
    assert pred.shape == (8,)


def _grid_mfs(n_inputs: int = 2, n_mfs: int = 2) -> dict[str, list]:  # type: ignore[type-arg]
    from highfis.memberships import GaussianMF

    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def _reference_grid_search(
    model,
    x,
    y,
    x_eval,
    y_eval,
    zeta_lambda,
    zeta_theta,
    use_lse,  # type: ignore[no-untyped-def]
):
    """The pre-refactor algorithm: one full ``deepcopy(model)`` per grid point.

    Kept verbatim as the numerical oracle for the snapshot/restore implementation.
    Returns every candidate's score, not just the winner, so parity is checked
    point-by-point rather than only at the argmax.
    """
    import copy

    scores: list[float] = []
    best = (float("-inf"), 0.0, 0.0, 0.0, 0.0)
    for zeta_l in zeta_lambda:
        for zeta_t in zeta_theta:
            candidate = copy.deepcopy(model)
            if use_lse:
                if "ZeroOrder" in candidate.consequent_layer.__class__.__name__:
                    candidate.convert_to_first_order()
                tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                candidate.apply_thresholds(tau_l, tau_t)
                candidate._fit_first_order_consequents_lse(x, y)
            else:
                tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                candidate.apply_thresholds(tau_l, tau_t)
            score = candidate._evaluate_threshold_score(x_eval, y_eval)
            scores.append(score)
            if score > best[0]:
                best = (score, tau_l, tau_t, zeta_l, zeta_t)
    return scores, best


def _spread_gates(model):  # type: ignore[no-untyped-def]
    """Give the gates a real spread so different zetas prune different subsets.

    A freshly built model has near-identical gate values, which makes every threshold
    degenerate (``zeta=0`` prunes everything, every point ties) and hides restore bugs.
    """
    with torch.no_grad():
        lam = model.rule_layer.lambda_gates
        lam.copy_(torch.linspace(0.2, 1.4, lam.numel()).reshape(lam.shape))
        theta = model.consequent_layer.theta_gates
        theta.copy_(torch.linspace(0.3, 1.5, theta.numel()).reshape(theta.shape))


@pytest.mark.parametrize("use_lse", [False, True])
def test_run_threshold_grid_search_matches_deepcopy_reference(use_lse: bool) -> None:
    """Snapshot/restore must pick the same winner as the per-point deepcopy it replaced."""
    from highfis.models._common import _run_threshold_grid_search
    from highfis.models._dg_aletsk import DGALETSKRegressorModel

    # zeta=0 maps to tau=max, which prunes every gate and makes the whole grid tie;
    # these values prune genuinely different subsets.
    zeta_lambda, zeta_theta = [0.25, 0.5, 0.75], [0.5, 1.0]
    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_grid_mfs())
    _spread_gates(model)
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]

    torch.manual_seed(1)
    ref_scores, (ref_score, ref_tau_l, ref_tau_t, ref_zeta_l, ref_zeta_t) = _reference_grid_search(
        model, x, y, x, y, zeta_lambda, zeta_theta, use_lse
    )

    scores: list[float] = []
    original = type(model)._evaluate_threshold_score

    def spy(self, x_eval, y_eval):  # type: ignore[no-untyped-def]
        score = original(self, x_eval, y_eval)
        scores.append(score)
        return score

    type(model)._evaluate_threshold_score = spy  # type: ignore[method-assign]
    try:
        torch.manual_seed(1)
        score, _state, tau_l, tau_t, zeta_l, zeta_t = _run_threshold_grid_search(
            model, x, y, x, y, zeta_lambda, zeta_theta, use_lse, False
        )
    finally:
        type(model)._evaluate_threshold_score = original  # type: ignore[method-assign]

    assert scores == pytest.approx(ref_scores)
    assert score == pytest.approx(ref_score)
    assert tau_l == pytest.approx(ref_tau_l)
    assert tau_t == pytest.approx(ref_tau_t)
    assert (zeta_l, zeta_t) == (ref_zeta_l, ref_zeta_t)


@pytest.mark.parametrize("use_lse", [False, True])
def test_run_threshold_grid_search_restores_state_between_points(use_lse: bool) -> None:
    """Every grid point must start from the same pristine gates.

    Without the per-point restore the points chain: the first one prunes, the next scores
    an already-pruned model, and the grid compares incomparable candidates.
    """
    from highfis.models._common import _run_threshold_grid_search
    from highfis.models._dg_aletsk import DGALETSKRegressorModel

    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_grid_mfs())
    _spread_gates(model)
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]

    seen: list[torch.Tensor] = []
    original = type(model).compute_thresholds

    def spy(self, zeta_l, zeta_t):  # type: ignore[no-untyped-def]
        seen.append(self.get_feature_gate_values().clone())
        return original(self, zeta_l, zeta_t)

    type(model).compute_thresholds = spy  # type: ignore[method-assign]
    try:
        _run_threshold_grid_search(model, x, y, x, y, [0.25, 0.5, 0.75], [0.5, 1.0], use_lse, False)
    finally:
        type(model).compute_thresholds = original  # type: ignore[method-assign]

    assert len(seen) == 6
    for i, gates in enumerate(seen[1:], start=1):
        assert torch.equal(gates, seen[0]), f"grid point {i} started from a mutated model"


@pytest.mark.parametrize("use_lse", [False, True])
@pytest.mark.parametrize("consequent_batch_norm", [False, True])
def test_run_threshold_grid_search_leaves_model_untouched(use_lse: bool, consequent_batch_norm: bool) -> None:
    """The grid must not mutate ``model``.

    ``search_thresholds`` derives the surviving feature/rule indices from the model's gate
    values *after* the grid returns, so a leaked pruned gate would silently change them.
    The batch-norm case also guards the running stats, which the grid's forward passes
    would otherwise update.
    """
    from highfis.models._common import _run_threshold_grid_search
    from highfis.models._dg_aletsk import DGALETSKRegressorModel

    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_grid_mfs(), consequent_batch_norm=consequent_batch_norm)
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    was_training = model.training

    _run_threshold_grid_search(model, x, y, x, y, [0.0, 0.5], [0.0, 0.5], use_lse, False)

    after = model.state_dict()
    assert set(after) == set(before)
    for k, expected in before.items():
        assert torch.equal(after[k], expected), f"{k} leaked out of the threshold grid"
    assert model.training == was_training


def test_run_threshold_grid_search_restores_model_when_a_point_raises() -> None:
    """A failure mid-grid must not leave the caller's model pruned."""
    from highfis.models._common import _run_threshold_grid_search
    from highfis.models._dg_aletsk import DGALETSKRegressorModel

    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_grid_mfs())
    _spread_gates(model)
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]
    before = {k: v.clone() for k, v in model.state_dict().items()}

    calls = {"n": 0}
    original = type(model)._evaluate_threshold_score

    def boom(self, x_eval, y_eval):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("scoring blew up")
        return original(self, x_eval, y_eval)

    type(model)._evaluate_threshold_score = boom  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="scoring blew up"):
            _run_threshold_grid_search(model, x, y, x, y, [0.25, 0.5, 0.75], [0.5, 1.0], False, False)
    finally:
        type(model)._evaluate_threshold_score = original  # type: ignore[method-assign]

    after = model.state_dict()
    for k, expected in before.items():
        assert torch.equal(after[k], expected), f"{k} stayed pruned after the grid raised"


def test_run_threshold_grid_search_restores_training_mode() -> None:
    """The LSE refit calls ``self.eval()``, and ``training`` lives outside ``state_dict``.

    Only observable on an already-first-order model: the grid then works on the caller's
    model directly rather than on a private copy, so the flag it flips is the caller's.
    """
    from highfis.models._common import _run_threshold_grid_search
    from highfis.models._dg_aletsk import DGALETSKRegressorModel

    torch.manual_seed(0)
    model = DGALETSKRegressorModel(_grid_mfs())
    model.convert_to_first_order()
    _spread_gates(model)
    model.train()
    x = torch.randn(16, 2)
    y = x[:, 0] - x[:, 1]

    _run_threshold_grid_search(model, x, y, x, y, [0.25, 0.5], [0.5], True, False)

    assert model.training is True


def test_get_consequent_bias_completes_the_rule() -> None:
    """The consequent intercept is reachable through the public introspection API.

    A first-order rule is ``score_r^c(x) = b_{r,c} + sum_d w_{r,c,d} x_d``;
    ``get_consequent_weights`` alone returns only ``w``, so reconstructing a full rule
    used to require reaching into ``consequent_layer.bias``.
    """
    from highfis.models import HTSKClassifierModel, HTSKRegressorModel

    clf = HTSKClassifierModel(_grid_mfs(n_inputs=3, n_mfs=2), n_classes=4)
    weight = clf.get_consequent_weights()
    bias = clf.get_consequent_bias()
    assert weight is not None and bias is not None
    assert weight.shape[:2] == bias.shape  # (rules, classes) shared prefix
    raw_bias = clf.consequent_layer.bias
    assert isinstance(raw_bias, torch.Tensor)
    assert torch.equal(bias, raw_bias.detach())

    reg = HTSKRegressorModel(_grid_mfs(n_inputs=3, n_mfs=2))
    assert reg.get_consequent_bias() is not None
