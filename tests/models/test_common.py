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
