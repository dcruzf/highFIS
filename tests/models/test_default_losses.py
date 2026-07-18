"""Cross-family guard: every model's default loss matches its source paper.

The surveyed high-dimensional TSK papers train their classifiers with mean squared error
on one-hot targets (the classical Sugeno convention); only TSK/HTSK/LogTSK have no
paper-specified classification loss and keep cross-entropy. Regression is always MSE.

The effective loss is the ``default_criterion`` *class attribute* (the trainer calls
``model.default_criterion()``); this test reads it directly so a family cannot silently
drift back to the wrong objective.
"""

from __future__ import annotations

import inspect

import pytest
from torch import nn

import highfis.models as models

# Classifiers whose source paper does NOT specify a classification loss; cross-entropy is a
# deliberate highFIS choice for these (softmax defuzzification lineage / PyTSK toolbox).
_CROSS_ENTROPY_CLASSIFIERS = {
    "TSKClassifierModel",
    "HTSKClassifierModel",
    "LogTSKClassifierModel",
}


def _model_classes(suffix: str) -> list[type]:
    out = []
    for name in dir(models):
        obj = getattr(models, name)
        if inspect.isclass(obj) and name.endswith(suffix) and not name.startswith("Base"):
            out.append(obj)
    return out


@pytest.mark.parametrize("model_cls", _model_classes("ClassifierModel"), ids=lambda c: c.__name__)
def test_classifier_default_loss_matches_paper(model_cls: type) -> None:
    expected = nn.CrossEntropyLoss if model_cls.__name__ in _CROSS_ENTROPY_CLASSIFIERS else nn.MSELoss
    assert model_cls.default_criterion is expected, (
        f"{model_cls.__name__} default loss drifted: {model_cls.default_criterion.__name__}"
    )


@pytest.mark.parametrize("model_cls", _model_classes("RegressorModel"), ids=lambda c: c.__name__)
def test_regressor_default_loss_is_mse(model_cls: type) -> None:
    assert model_cls.default_criterion is nn.MSELoss, (
        f"{model_cls.__name__} regressor loss must be MSE, got {model_cls.default_criterion.__name__}"
    )


def test_guard_covers_every_family() -> None:
    """Fail loudly if the model enumeration silently returns nothing."""
    assert len(_model_classes("ClassifierModel")) >= 14
    assert len(_model_classes("RegressorModel")) >= 14
