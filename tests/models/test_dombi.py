from __future__ import annotations

import pytest
import torch
from torch import nn

from highfis.memberships import GaussianMF
from highfis.models import DombiTSKClassifierModel, DombiTSKRegressorModel


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_dombitsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = DombiTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_=2.0)
    x = torch.randn(6, 2)
    norm_w = model.forward_antecedents(x)
    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-06)


def test_dombitsk_classifier_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        DombiTSKClassifierModel(_build_input_mfs(), n_classes=2, lambda_=0.0)


def test_dombitsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DombiTSKClassifierModel(_build_input_mfs(), n_classes=1, lambda_=1.0)


def test_dombitsk_classifier_default_t_norm_fn_branch() -> None:
    model = DombiTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_=2.0)
    x = torch.randn(4, 2)
    logits = model.forward(x)
    assert logits.shape == (4, 2)


def test_dombi_tsk_regressor_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        DombiTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_=0.0)


def test_dombi_tsk_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DombiTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=1, lambda_=1.0)


def test_dombi_tsk_classifier_default_t_norm_fn_branch() -> None:
    model = DombiTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2, lambda_=1.5)
    x = torch.randn(4, 2)
    logits = model.forward(x)
    assert logits.shape == (4, 2)


def test_dombi_tsk_classifier_explicit_t_norm_fn_branch() -> None:
    model = DombiTSKClassifierModel(
        _build_input_mfs(n_inputs=2, n_mfs=2),
        n_classes=2,
        lambda_=1.5,
        t_norm=lambda terms, dim=-1: terms.prod(dim=dim),
    )
    x = torch.randn(4, 2)
    logits = model.forward(x)
    assert logits.shape == (4, 2)


def test_dombi_tsk_regressor_default_t_norm_fn_branch() -> None:
    model = DombiTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_=1.0)
    x = torch.randn(4, 2)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_dombi_tsk_regressor_explicit_t_norm_fn_branch() -> None:
    model = DombiTSKRegressorModel(
        _build_input_mfs(n_inputs=2, n_mfs=2), lambda_=1.0, t_norm=lambda terms, dim=-1: terms.prod(dim=dim)
    )
    x = torch.randn(4, 2)
    out = model.forward(x)
    assert out.shape == (4, 1)


def test_dombi_tsk_classifier_default_criterion() -> None:
    model = DombiTSKClassifierModel(_build_input_mfs(), n_classes=2, lambda_=1.0)
    assert isinstance(model._default_criterion(), nn.CrossEntropyLoss)


def test_dombi_tsk_regressor_default_criterion() -> None:
    model = DombiTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2), lambda_=1.0)
    assert isinstance(model._default_criterion(), nn.MSELoss)


def test_dombi_classifier_zero_init_branches() -> None:
    mfs = {"x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]}
    model_false = DombiTSKClassifierModel(mfs, n_classes=2, zero_consequent_init=False)
    assert model_false.zero_consequent_init is False

    class MockConsequent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = "not_a_tensor"
            self.bias = None

    model_mock = DombiTSKClassifierModel(mfs, n_classes=2, zero_consequent_init=True)
    model_mock.consequent_layer = MockConsequent()
    model_mock._zero_initialize_consequents()
