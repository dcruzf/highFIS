"""Tests for highfis.protocols."""

from __future__ import annotations

import torch
from torch import Tensor

from highfis.protocols import ConsequentFn, Defuzzifier, MembershipFn, TNorm


def _dummy_mf(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def _dummy_tnorm(terms: Tensor) -> Tensor:
    return torch.prod(terms, dim=1)


def _dummy_defuzz(w: Tensor) -> Tensor:
    return w / w.sum(dim=1, keepdim=True)


def _dummy_cons(x: Tensor, norm_w: Tensor) -> Tensor:
    return norm_w @ x.T


def test_membership_fn_protocol() -> None:
    assert isinstance(_dummy_mf, MembershipFn)


def test_tnorm_protocol() -> None:
    assert isinstance(_dummy_tnorm, TNorm)


def test_defuzzifier_protocol() -> None:
    assert isinstance(_dummy_defuzz, Defuzzifier)


def test_consequent_fn_protocol() -> None:
    assert isinstance(_dummy_cons, ConsequentFn)


def test_nn_module_satisfies_membership_fn() -> None:
    """nn.Module with __call__ accepting Tensor satisfies MembershipFn."""
    from highfis.memberships import GaussianMF

    mf = GaussianMF(mean=0.0, sigma=1.0)
    assert isinstance(mf, MembershipFn)
