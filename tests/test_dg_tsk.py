from __future__ import annotations

import pytest
import torch

from highfis.estimators import DGTSKClassifierEstimator, DGTSKRegressorEstimator
from highfis.memberships import GaussianMF
from highfis.models import DGTSKClassifier, DGTSKRegressor


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


def test_dgtsk_classifier_forward_shapes() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=3)
    x = torch.randn(5, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (5, 3)
    assert proba.shape == (5, 3)
    assert pred.shape == (5,)


def test_dgtsk_regressor_forward_shape() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(4, 2)

    output = model.forward(x)

    assert output.shape == (4, 1)


def test_dgtsk_classifier_architecture() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=3)

    assert model.rule_layer.__class__.__name__ == "DGTSKRuleLayer"
    assert model.consequent_layer.__class__.__name__ == "GatedClassificationZeroOrderConsequentLayer"


def test_dgtsk_classifier_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DGTSKClassifier(_build_input_mfs(), n_classes=1)


def test_dgtsk_estimator_instantiation() -> None:
    clf = DGTSKClassifierEstimator(n_mfs=2, mf_init="kmeans", use_en_frb=True)
    reg = DGTSKRegressorEstimator(n_mfs=2, mf_init="kmeans", use_en_frb=True)

    assert clf is not None
    assert reg is not None


def test_dgtsk_classifier_apply_thresholds_invalid_thresholds_raises() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    with pytest.raises(ValueError, match="thresholds must be finite"):
        model.apply_thresholds(float("nan"), 0.0)


def test_dgtsk_classifier_search_thresholds_verbose_and_inplace_true() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        x_val=x,
        y_val=y,
        use_lse=False,
        inplace=True,
        verbose=True,
    )

    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}
    assert 0.0 <= result["best_score"] <= 1.0


def test_dgtsk_regressor_search_thresholds_verbose_and_inplace_true() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(
        x,
        y,
        zeta_lambda=[0.0, 1.0],
        zeta_theta=[0.0, 1.0],
        x_val=x,
        y_val=y,
        use_lse=False,
        inplace=True,
        verbose=True,
    )

    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgtsk_classifier_convert_to_first_order_preserves_theta() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    theta_before = model.consequent_layer.theta_gates.detach().clone()

    model.convert_to_first_order()

    assert model.consequent_layer.__class__.__name__ == "GatedClassificationConsequentLayer"
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_dgtsk_regressor_convert_to_first_order_preserves_theta() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    theta_before = model.consequent_layer.theta_gates.detach().clone()

    model.convert_to_first_order()

    assert model.consequent_layer.__class__.__name__ == "GatedRegressionConsequentLayer"
    assert torch.allclose(model.consequent_layer.theta_gates.detach(), theta_before)


def test_dgtsk_classifier_thresholds_and_pruning() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    model.rule_layer.lambda_gates.data.fill_(0.0)
    model.rule_layer.lambda_gates.data[0, 0] = 1.0
    model.consequent_layer.theta_gates.data.fill_(0.0)
    model.consequent_layer.theta_gates.data[0] = 1.0

    tau_lambda, tau_theta = model.compute_thresholds(0.5, 0.5)
    assert torch.isclose(torch.tensor(tau_lambda), torch.tensor(0.5))
    assert torch.isclose(torch.tensor(tau_theta), torch.tensor(0.5))

    model.apply_thresholds(tau_lambda, tau_theta)
    assert model.rule_layer.lambda_gates.data[0, 0] == 1.0
    assert model.consequent_layer.theta_gates.data[0] == 1.0


def test_dgtsk_classifier_search_thresholds_returns_result() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=5, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(x, y, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], inplace=False)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}
    assert 0.0 <= result["best_score"] <= 1.0


def test_dgtsk_regressor_search_thresholds_returns_result() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)
    model.fit_dg_phase(x, y, epochs=5, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(x, y, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], inplace=False)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgtsk_classifier_search_thresholds_default_zeta_lists() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(x, y, zeta_lambda=None, zeta_theta=None, inplace=False)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgtsk_regressor_search_thresholds_default_zeta_lists() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)
    model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)

    result = model.search_thresholds(x, y, zeta_lambda=None, zeta_theta=None, inplace=False)
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgtsk_classifier_search_thresholds_on_first_order_model() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))

    result = model.search_thresholds(x, y, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], inplace=True, verbose=False)
    assert model.consequent_layer.__class__.__name__ == "GatedClassificationConsequentLayer"
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgtsk_regressor_search_thresholds_on_first_order_model() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    x = torch.randn(20, 2)
    y = torch.randn(20)

    result = model.search_thresholds(x, y, zeta_lambda=[0.0, 1.0], zeta_theta=[0.0, 1.0], inplace=True, verbose=False)
    assert model.consequent_layer.__class__.__name__ == "GatedRegressionConsequentLayer"
    assert set(result) >= {"best_score", "best_zeta_lambda", "best_zeta_theta", "tau_lambda", "tau_theta"}


def test_dgtsk_classifier_convert_to_first_order_idempotent() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    model.convert_to_first_order()
    model.convert_to_first_order()
    assert model.consequent_layer.__class__.__name__ == "GatedClassificationConsequentLayer"


def test_dgtsk_regressor_convert_to_first_order_idempotent() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    model.convert_to_first_order()
    model.convert_to_first_order()
    assert model.consequent_layer.__class__.__name__ == "GatedRegressionConsequentLayer"


def test_dgtsk_regressor_search_thresholds_no_candidates_raises() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(8, 2)
    y = torch.randn(8)

    with pytest.raises(RuntimeError, match="threshold search did not yield a valid candidate"):
        model.search_thresholds(x, y, zeta_lambda=[], zeta_theta=[], inplace=False)


def test_dgtsk_classifier_search_thresholds_no_candidates_raises() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,))

    with pytest.raises(RuntimeError, match="threshold search did not yield a valid candidate"):
        model.search_thresholds(x, y, zeta_lambda=[], zeta_theta=[], inplace=False)


def test_dgtsk_regressor_apply_thresholds_invalid_thresholds_raises() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))

    with pytest.raises(ValueError, match="thresholds must be finite"):
        model.apply_thresholds(float("nan"), 0.0)


def test_dgtsk_classifier_fit_first_order_consequents_requires_conversion() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,))

    with pytest.raises(ValueError, match=r"convert_to_first_order\(\) must be called before LSE consequent fitting"):
        model._fit_first_order_consequents_lse(x, y)


def test_dgtsk_regressor_fit_first_order_consequents_requires_conversion() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(8, 2)
    y = torch.randn(8)

    with pytest.raises(ValueError, match=r"convert_to_first_order\(\) must be called before LSE consequent fitting"):
        model._fit_first_order_consequents_lse(x, y)


def test_dgtsk_classifier_fit_dg_phase_and_finetune() -> None:
    model = DGTSKClassifier(_build_input_mfs(), n_classes=2)
    x = torch.randn(16, 3)
    y = torch.randint(0, 2, (16,))

    history_dg = model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)
    assert isinstance(history_dg, dict)

    history_ft = model.fit_finetune(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)
    assert isinstance(history_ft, dict)


def test_dgtsk_regressor_fit_dg_phase_and_finetune() -> None:
    model = DGTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
    x = torch.randn(20, 2)
    y = torch.randn(20)

    history_dg = model.fit_dg_phase(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)
    assert isinstance(history_dg, dict)

    history_ft = model.fit_finetune(x, y, epochs=2, learning_rate=1e-2, batch_size=8, shuffle=False)
    assert isinstance(history_ft, dict)
