from __future__ import annotations

import pytest
import torch
from torch import nn

from highfis.memberships import GaussianMF
from highfis.models import (
    HDFISMinClassifierModel,
    HDFISMinRegressorModel,
    HDFISProdClassifierModel,
    HDFISProdRegressorModel,
)
from highfis.models._hdfis import _zero_initialize_consequents as _hdfis_zero_initialize_consequents


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_hdfismin_classifier_freezes_membership_parameters() -> None:
    model = HDFISMinClassifierModel(_build_input_mfs(), n_classes=2)
    assert all(not p.requires_grad for p in model.membership_layer.parameters())


def test_hdfisprod_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        HDFISProdClassifierModel(_build_input_mfs(), n_classes=1)


def test_hdfismin_classifier_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        HDFISMinClassifierModel(_build_input_mfs(), n_classes=1)


def test_hdfismin_regressor_freezes_membership_parameters() -> None:
    model = HDFISMinRegressorModel(_build_input_mfs(), rule_base="coco")
    assert all(not p.requires_grad for p in model.membership_layer.parameters())


def test_hdfis_models_can_zero_initialize_consequents() -> None:
    clf = HDFISProdClassifierModel(_build_input_mfs(), n_classes=2, zero_consequent_init=True)
    reg = HDFISProdRegressorModel(_build_input_mfs(), zero_consequent_init=True)
    min_clf = HDFISMinClassifierModel(_build_input_mfs(), n_classes=2, zero_consequent_init=True)
    min_reg = HDFISMinRegressorModel(_build_input_mfs(), zero_consequent_init=True)
    clf_weight = getattr(clf.consequent_layer, "weight", None)
    clf_bias = getattr(clf.consequent_layer, "bias", None)
    reg_weight = getattr(reg.consequent_layer, "weight", None)
    reg_bias = getattr(reg.consequent_layer, "bias", None)
    min_clf_weight = getattr(min_clf.consequent_layer, "weight", None)
    min_clf_bias = getattr(min_clf.consequent_layer, "bias", None)
    min_reg_weight = getattr(min_reg.consequent_layer, "weight", None)
    min_reg_bias = getattr(min_reg.consequent_layer, "bias", None)
    assert isinstance(clf_weight, torch.Tensor)
    assert isinstance(clf_bias, torch.Tensor)
    assert isinstance(reg_weight, torch.Tensor)
    assert isinstance(reg_bias, torch.Tensor)
    assert isinstance(min_clf_weight, torch.Tensor)
    assert isinstance(min_clf_bias, torch.Tensor)
    assert isinstance(min_reg_weight, torch.Tensor)
    assert isinstance(min_reg_bias, torch.Tensor)
    assert torch.allclose(clf_weight, torch.zeros_like(clf_weight))
    assert torch.allclose(clf_bias, torch.zeros_like(clf_bias))
    assert torch.allclose(reg_weight, torch.zeros_like(reg_weight))
    assert torch.allclose(reg_bias, torch.zeros_like(reg_bias))
    assert torch.allclose(min_clf_weight, torch.zeros_like(min_clf_weight))
    assert torch.allclose(min_clf_bias, torch.zeros_like(min_clf_bias))
    assert torch.allclose(min_reg_weight, torch.zeros_like(min_reg_weight))
    assert torch.allclose(min_reg_bias, torch.zeros_like(min_reg_bias))


def test_hdfis_zero_initialize_consequents_handles_missing_params() -> None:
    class DummyLayer(nn.Module):
        pass

    _hdfis_zero_initialize_consequents(DummyLayer())


def test_hdfis_default_criteria() -> None:
    clf_p = HDFISProdClassifierModel(_build_input_mfs(), n_classes=2)
    reg_p = HDFISProdRegressorModel(_build_input_mfs())
    clf_m = HDFISMinClassifierModel(_build_input_mfs(), n_classes=2)
    reg_m = HDFISMinRegressorModel(_build_input_mfs())

    # Paper-faithful: both HDFIS classifiers use MSE on one-hot targets (eq. 14).
    assert isinstance(clf_p.default_criterion(), nn.MSELoss)
    assert isinstance(reg_p.default_criterion(), nn.MSELoss)
    assert isinstance(clf_m.default_criterion(), nn.MSELoss)
    assert isinstance(reg_m.default_criterion(), nn.MSELoss)
