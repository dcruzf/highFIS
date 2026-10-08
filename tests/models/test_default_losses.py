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
from highfis.losses import HalfSumSquaredErrorLoss, SumSquaredErrorLoss

# Classifiers whose source paper does NOT specify a classification loss; cross-entropy is a
# deliberate highFIS choice for these (softmax defuzzification lineage / PyTSK toolbox).
_CROSS_ENTROPY_CLASSIFIERS = {
    "TSKClassifierModel",
    "HTSKClassifierModel",
    "LogTSKClassifierModel",
}


# Families trained by plain gradient descent whose article (or reference code) fixes the scale
# of the squared error: summed over the outputs, averaged over the samples and halved.
_HALF_SUM_SQUARED_ERROR = {
    "DGTSKClassifierModel",
    "DGTSKRegressorModel",
}


# FSRE-ADATSK: the same, without the one half (the gradients after Eq. (8) of its article).
_SUM_SQUARED_ERROR = {"FSREADATSKClassifierModel"}


def _model_classes(suffix: str) -> list[type]:
    out = []
    for name in dir(models):
        obj = getattr(models, name)
        if inspect.isclass(obj) and name.endswith(suffix) and not name.startswith("Base"):
            out.append(obj)
    return out


@pytest.mark.parametrize("model_cls", _model_classes("ClassifierModel"), ids=lambda c: c.__name__)
def test_classifier_default_loss_matches_paper(model_cls: type) -> None:
    if model_cls.__name__ in _CROSS_ENTROPY_CLASSIFIERS:
        expected: type = nn.CrossEntropyLoss
    elif model_cls.__name__ in _HALF_SUM_SQUARED_ERROR:
        expected = HalfSumSquaredErrorLoss
    elif model_cls.__name__ in _SUM_SQUARED_ERROR:
        expected = SumSquaredErrorLoss
    else:
        expected = nn.MSELoss
    assert model_cls.default_criterion is expected, (
        f"{model_cls.__name__} default loss drifted: {model_cls.default_criterion.__name__}"
    )


@pytest.mark.parametrize("model_cls", _model_classes("RegressorModel"), ids=lambda c: c.__name__)
def test_regressor_default_loss_is_mse(model_cls: type) -> None:
    expected = HalfSumSquaredErrorLoss if model_cls.__name__ in _HALF_SUM_SQUARED_ERROR else nn.MSELoss
    assert model_cls.default_criterion is expected, (
        f"{model_cls.__name__} regressor loss must be {expected.__name__}, got {model_cls.default_criterion.__name__}"
    )


def test_half_sum_squared_error_is_the_sum_over_outputs_halved() -> None:
    """sum((y - z) ** 2) / (2N): MSELoss times C / 2 for C outputs."""
    import torch

    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    output = torch.tensor([[0.5, 0.2, 0.1], [0.1, 0.7, 0.0]])

    loss = HalfSumSquaredErrorLoss()(output, target)

    assert loss.item() == pytest.approx(((target - output) ** 2).sum().item() / 4.0)
    assert loss.item() == pytest.approx(nn.MSELoss()(output, target).item() * 3.0 / 2.0)
    assert isinstance(HalfSumSquaredErrorLoss(), nn.MSELoss)


def test_guard_covers_every_family() -> None:
    """Fail loudly if the model enumeration silently returns nothing."""
    assert len(_model_classes("ClassifierModel")) >= 14
    assert len(_model_classes("RegressorModel")) >= 14


def test_sum_squared_error_is_the_sum_over_outputs() -> None:
    """sum((y - z) ** 2) / N: MSELoss times C for C outputs, and equal to it for one output."""
    import torch

    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    output = torch.tensor([[0.5, 0.2, 0.1], [0.1, 0.7, 0.0]])

    loss = SumSquaredErrorLoss()(output, target)

    assert loss.item() == pytest.approx(((target - output) ** 2).sum().item() / 2.0)
    assert loss.item() == pytest.approx(nn.MSELoss()(output, target).item() * 3.0)
    one_output = SumSquaredErrorLoss()(output[:, 0], target[:, 0]).item()
    assert one_output == pytest.approx(nn.MSELoss()(output[:, 0], target[:, 0]).item())
