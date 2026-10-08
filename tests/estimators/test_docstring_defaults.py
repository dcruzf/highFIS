"""The defaults and optimizers named in estimator docstrings must match the code.

The "Args" text of the constructors drifted from the signatures once; this guard reads
both and fails when a stated default differs from the real one, or when the
``learning_rate`` description names an optimizer the family does not train with.
"""

from __future__ import annotations

import inspect
import re

import pytest

import highfis

ESTIMATORS: list[str] = sorted(n for n in highfis.__all__ if n.endswith(("Classifier", "Regressor")))
ARGS = ("epochs", "dg_epochs", "finetune_epochs", "learning_rate", "n_mfs", "batch_size", "patience", "weight_decay")

# highfis/optim/_utils.py::_select_optimizer_class and the models' ``_optimizer_type``.
OPTIMIZER = {
    "ADATSK": "SGD",
    "FSREADATSK": "SGD",
    "DGTSK": "SGD",
    "ADPTSK": "Adam",
    "DombiTSK": "Adam",
    "ADMTSK": "Adam",
    "AYATSK": "Adam",
    "DGALETSK": "Adam",
}


def _optimizer(name: str) -> str:
    return OPTIMIZER.get(re.sub(r"(Classifier|Regressor)$", "", name), "AdamW")


def _described(doc: str, arg: str) -> str | None:
    """Text of the ``arg:`` entry in a Google-style "Args" section."""
    match = re.search(rf"^\s*{arg}:(.*?)(?=^\s*\w+:|\Z)", doc, re.S | re.M)
    return match.group(1) if match else None


def _same(stated: str, real: object) -> bool:
    try:
        return float(stated) == float(real)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(real) == stated


def _mismatches(name: str) -> list[str]:
    cls = getattr(highfis, name)
    signature = inspect.signature(cls.__init__).parameters
    doc = inspect.getdoc(cls.__init__) or ""
    found = []
    for arg in ARGS:
        text = _described(doc, arg) if arg in signature else None
        if text is None:
            continue
        stated = re.search(r"[Dd]efault[s]?\s*(?:is|to|:)?\s*``([^`]+)``", text)
        if stated and not _same(stated.group(1).strip().strip('"'), signature[arg].default):
            found.append(f"{arg}: docstring says {stated.group(1)}, signature has {signature[arg].default!r}")
        named = re.search(r"\b(AdamW|Adam|SGD)\b", text) if arg == "learning_rate" else None
        if named and named.group(1) != _optimizer(name):
            found.append(f"learning_rate: docstring says {named.group(1)}, the family trains with {_optimizer(name)}")
    return found


@pytest.mark.parametrize("name", ESTIMATORS)
def test_docstring_defaults_match_the_signature(name: str) -> None:
    assert _mismatches(name) == []


def test_the_optimizer_table_matches_the_trainer() -> None:
    """The table above is only a guard if it agrees with what the trainer really selects."""
    import numpy as np

    from highfis.optim._utils import _select_optimizer_class

    rng = np.random.default_rng(0)
    x = rng.random((30, 3))
    for name in ESTIMATORS:
        cls = getattr(highfis, name)
        epochs = {p: 1 for p in inspect.signature(cls.__init__).parameters if p.endswith("epochs")}
        y = (x[:, 0] > 0.5).astype(int) if name.endswith("Classifier") else x[:, 0]
        model = cls(random_state=0, **epochs).fit(x, y).model_
        assert _select_optimizer_class(model).__name__ == _optimizer(name), name
