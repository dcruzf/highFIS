"""Cross-estimator guards for the scikit-learn parameter contract.

``get_params``/``set_params`` are inherited from ``BaseEstimator``, which derives the
parameter names by introspecting the *concrete* class's ``__init__``. Two mistakes follow
from that and neither is caught by the per-estimator tests:

* a constructor argument that ``__init__`` accepts but never stores (or forwards to
  ``super().__init__``) silently reverts to the base default, so ``clone`` -- and
  therefore ``GridSearchCV``, ``cross_val_score`` and ``Pipeline`` -- searches a no-op
  axis;
* a parameter declared only on the base class is invisible to ``get_params`` and is
  dropped by ``clone``.

These tests sweep every exported estimator so a new parameter cannot be added to some
constructors and forgotten in others.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from sklearn.base import clone

import highfis

ESTIMATORS: list[str] = sorted(n for n in highfis.__all__ if n.endswith(("Classifier", "Regressor")))


class _Sentinel:
    """A value no constructor can confuse with a default, of no particular type.

    Probing with a typed value (an int for an int param, and nothing at all for a ``str``)
    leaves holes: it silently skipped ``device`` and let ``LogTSKClassifier(device="cuda")``
    train on the CPU. The estimators store constructor arguments verbatim and validate at
    ``fit`` time, so an opaque object round-trips through ``get_params`` exactly like a real
    one -- which makes this a test of the plumbing rather than of the types.
    """

    _next_tag = 0

    def __init__(self) -> None:
        _Sentinel._next_tag += 1
        self.tag = _Sentinel._next_tag

    def __eq__(self, other: object) -> bool:
        # ``clone`` deep-copies every parameter value, so equality has to survive a copy.
        return isinstance(other, _Sentinel) and other.tag == self.tag

    def __hash__(self) -> int:
        return hash(self.tag)

    def __repr__(self) -> str:
        return f"<sentinel {self.tag}>"


def _declared_params(name: str) -> list[tuple[str, Any]]:
    cls = getattr(highfis, name)
    out = []
    for param, spec in inspect.signature(cls.__init__).parameters.items():
        if param == "self" or spec.default is inspect.Parameter.empty:
            continue
        out.append((param, spec.default))
    return out


def test_estimators_are_discovered() -> None:
    """Guard the sweep itself: an empty list would make every test below vacuous."""
    assert len(ESTIMATORS) >= 28


@pytest.mark.parametrize("name", ESTIMATORS)
def test_constructor_params_round_trip_through_get_params(name: str) -> None:
    """Every constructor argument must survive ``get_params``.

    A dropped argument means ``clone`` silently resets it, so a ``GridSearchCV`` over that
    parameter would search a no-op axis and report a flat curve.
    """
    cls = getattr(highfis, name)
    dropped = []
    for param, _default in _declared_params(name):
        probe = _Sentinel()
        got = cls(**{param: probe}).get_params().get(param, "<missing>")
        if got is not probe:
            dropped.append(f"{param}: passed the sentinel, get_params returned {got!r}")
    assert not dropped, f"{name} drops constructor arguments: " + "; ".join(dropped)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_constructor_params_are_visible_to_get_params(name: str) -> None:
    """Every constructor argument must appear in ``get_params``, or ``clone`` drops it."""
    cls = getattr(highfis, name)
    params = cls().get_params()
    missing = [p for p, _ in _declared_params(name) if p not in params]
    assert not missing, f"{name} hides constructor arguments from get_params: {missing}"


@pytest.mark.parametrize("name", ESTIMATORS)
def test_clone_preserves_non_default_params(name: str) -> None:
    """``clone`` must reproduce non-default parameters; this is what cross-validation relies on."""
    cls = getattr(highfis, name)
    overrides = {param: _Sentinel() for param, _ in _declared_params(name)}
    cloned = clone(cls(**overrides))
    for param, expected in overrides.items():
        assert cloned.get_params()[param] == expected, f"{name}.{param} lost by clone()"
