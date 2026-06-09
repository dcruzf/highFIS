from __future__ import annotations

import pytest
import torch

from highfis.memberships import GaussianMF
from highfis.models import HTSKClassifierModel, HTSKRegressorModel


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_htsk_classifier_init_validates_arguments() -> None:
    with pytest.raises(ValueError, match="input_mfs must not be empty"):
        HTSKClassifierModel({}, n_classes=2)
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        HTSKClassifierModel(_build_input_mfs(), n_classes=1)


def test_htsk_classifier_forward_predict_shapes() -> None:
    model = HTSKClassifierModel(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)
    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)
    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-06)


def test_htsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = HTSKClassifierModel(_build_input_mfs(), n_classes=2)
    x = torch.randn(6, 3)
    norm_w = model.forward_antecedents(x)
    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-06)


def test_htsk_regressor_init_validates_arguments() -> None:
    with pytest.raises(ValueError, match="input_mfs must not be empty"):
        HTSKRegressorModel({})


def test_htsk_regressor_forward_predict_shapes() -> None:
    model = HTSKRegressorModel(_build_input_mfs())
    x = torch.randn(8, 3)
    out = model.forward(x)
    pred = model.predict(x)
    assert out.shape == (8, 1)
    assert pred.shape == (8,)


def test_htsk_regressor_forward_antecedents_row_sum_one() -> None:
    model = HTSKRegressorModel(_build_input_mfs())
    x = torch.randn(6, 3)
    norm_w = model.forward_antecedents(x)
    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-06)


def test_htsk_default_criteria() -> None:
    from torch import nn

    from highfis.models import TSKClassifierModel, TSKRegressorModel

    clf_h = HTSKClassifierModel(_build_input_mfs(), n_classes=2)
    reg_h = HTSKRegressorModel(_build_input_mfs())
    clf_t = TSKClassifierModel(_build_input_mfs(), n_classes=2)
    reg_t = TSKRegressorModel(_build_input_mfs())

    assert isinstance(clf_h._default_criterion(), nn.CrossEntropyLoss)
    assert isinstance(reg_h._default_criterion(), nn.MSELoss)
    assert isinstance(clf_t._default_criterion(), nn.CrossEntropyLoss)
    assert isinstance(reg_t._default_criterion(), nn.MSELoss)
