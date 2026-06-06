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
