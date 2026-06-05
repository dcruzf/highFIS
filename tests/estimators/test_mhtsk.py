from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from highfis import MHTSKClassifier, MHTSKRegressor
from highfis.estimators._base import (
    _build_mhtsk_input_mfs,
    _extract_mhtsk_rule_indices,
    _resolve_mhtsk_scale_parameters,
    feature_coverage_rate,
)
from highfis.estimators._mhtsk import (
    _MHTSKPaperStrictTrainer,
    _resolve_mhtsk_paper_strict_classifier_config,
    _resolve_mhtsk_paper_strict_regressor_config,
    _strict_mhtsk_scale_from_dimension,
)
from highfis.memberships import GaussianMF
from highfis.models import MHTSKClassifierModel


def _make_dataset(n_samples: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n_samples, 3)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0.0).astype(int)
    return x, y


def test_mhtsk_classifier_estimator_fit_predict() -> None:
    x, y = _make_dataset(40)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (x.shape[0],)


def test_mhtsk_regressor_estimator_fit_predict() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (x.shape[0],)


def test_mhtsk_classifier_estimator_rule_extraction_reduces_rules() -> None:
    x, y = _make_dataset(40)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        crcr_s=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)

    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0
    assert est.model_.n_rules == len(est._extracted_rule_indices_)


def test_mhtsk_regressor_estimator_rule_extraction_reduces_rules() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)

    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0
    assert est.model_.n_rules == len(est._extracted_rule_indices_)


def test_mhtsk_paper_strict_defaults() -> None:
    clf = MHTSKClassifier(paper_strict=True)
    reg = MHTSKRegressor(paper_strict=True)

    assert clf.n_mfs == 3
    assert clf.fcm_m == 2.0
    assert clf.rule_sigma == 1.0
    assert clf.xi == 743.0
    assert clf.instance_sample_fraction == 0.8
    assert clf.rule_extraction is True
    assert clf.crcr_us == 0.5
    assert clf.crcr_s == 0.5
    assert clf.retrain_after_extraction is True

    assert reg.n_mfs == 3
    assert reg.fcm_m == 2.0
    assert reg.rule_sigma == 1.0
    assert reg.xi == 743.0
    assert reg.instance_sample_fraction == 0.8
    assert reg.rule_extraction is True
    assert reg.crcr_us == 0.5
    assert reg.retrain_after_extraction is True


def test_mhtsk_paper_strict_rejects_conflicting_hyperparameters() -> None:
    with pytest.raises(ValueError, match="paper_strict requires n_mfs=3"):
        MHTSKClassifier(paper_strict=True, n_mfs=4)
    with pytest.raises(ValueError, match="paper_strict requires n_mfs=3"):
        MHTSKRegressor(paper_strict=True, n_mfs=4)
    with pytest.raises(ValueError, match="paper_strict computes n_heads from input dimension"):
        MHTSKClassifier(paper_strict=True, n_heads=100)
    with pytest.raises(ValueError, match="paper_strict computes n_heads from input dimension"):
        MHTSKRegressor(paper_strict=True, n_heads=100)
    with pytest.raises(ValueError, match="paper_strict computes head_size from input dimension"):
        MHTSKClassifier(paper_strict=True, head_size=10)
    with pytest.raises(ValueError, match="paper_strict computes head_size from input dimension"):
        MHTSKRegressor(paper_strict=True, head_size=10)
    with pytest.raises(ValueError, match="paper_strict requires fcr_target=None"):
        MHTSKClassifier(paper_strict=True, fcr_target=0.9)
    with pytest.raises(ValueError, match="paper_strict requires fcr_target=None"):
        MHTSKRegressor(paper_strict=True, fcr_target=0.9)
    with pytest.raises(ValueError, match="paper_strict requires h_value=None"):
        MHTSKClassifier(paper_strict=True, h_value=3.0)
    with pytest.raises(ValueError, match="paper_strict requires h_value=None"):
        MHTSKRegressor(paper_strict=True, h_value=3.0)
    with pytest.raises(ValueError, match="paper_strict requires rule_extraction=True"):
        MHTSKClassifier(paper_strict=True, rule_extraction=False)
    with pytest.raises(ValueError, match="paper_strict requires rule_extraction=True"):
        MHTSKRegressor(paper_strict=True, rule_extraction=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"head_size_ratio": 0.1}, "paper_strict requires head_size_ratio=None"),
        ({"fcm_m": 2.1}, "paper_strict requires fcm_m=2.0"),
        ({"rule_sigma": 1.1}, "paper_strict requires rule_sigma=1.0"),
        ({"xi": 700.0}, "paper_strict requires xi=743.0"),
        ({"instance_sample_fraction": 0.7}, "paper_strict requires instance_sample_fraction=0.8"),
        ({"crcr_us": 0.6}, "paper_strict requires crcr_us=0.5"),
        ({"crcr_s": 0.6}, "paper_strict requires crcr_s=0.5"),
        ({"retrain_after_extraction": False}, "paper_strict requires retrain_after_extraction=True"),
    ],
)
def test_mhtsk_paper_strict_classifier_config_rejects_invalid_values(
    kwargs: dict[str, float | bool],
    message: str,
) -> None:
    base_kwargs: dict[str, float | bool | None] = {
        "n_mfs": None,
        "n_heads": None,
        "head_size": None,
        "head_size_ratio": None,
        "fcm_m": None,
        "rule_sigma": None,
        "fcr_target": None,
        "h_value": None,
        "xi": None,
        "instance_sample_fraction": None,
        "rule_extraction": None,
        "crcr_us": None,
        "crcr_s": None,
        "retrain_after_extraction": None,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _resolve_mhtsk_paper_strict_classifier_config(
            paper_strict=True,
            **base_kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"head_size_ratio": 0.1}, "paper_strict requires head_size_ratio=None"),
        ({"fcm_m": 2.1}, "paper_strict requires fcm_m=2.0"),
        ({"rule_sigma": 1.1}, "paper_strict requires rule_sigma=1.0"),
        ({"xi": 700.0}, "paper_strict requires xi=743.0"),
        ({"instance_sample_fraction": 0.7}, "paper_strict requires instance_sample_fraction=0.8"),
        ({"crcr_us": 0.6}, "paper_strict requires crcr_us=0.5"),
        ({"retrain_after_extraction": False}, "paper_strict requires retrain_after_extraction=True"),
    ],
)
def test_mhtsk_paper_strict_regressor_config_rejects_invalid_values(
    kwargs: dict[str, float | bool],
    message: str,
) -> None:
    base_kwargs: dict[str, float | bool | None] = {
        "n_mfs": None,
        "n_heads": None,
        "head_size": None,
        "head_size_ratio": None,
        "fcm_m": None,
        "rule_sigma": None,
        "fcr_target": None,
        "h_value": None,
        "xi": None,
        "instance_sample_fraction": None,
        "rule_extraction": None,
        "crcr_us": None,
        "retrain_after_extraction": None,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _resolve_mhtsk_paper_strict_regressor_config(
            paper_strict=True,
            **base_kwargs,
        )


def test_mhtsk_paper_strict_scale_policy_helper() -> None:
    assert _strict_mhtsk_scale_from_dimension(1000, sigma=1.0, xi=743.0) == (20, 200)
    assert _strict_mhtsk_scale_from_dimension(6000, sigma=1.0, xi=743.0) == (60, 300)


def test_mhtsk_classifier_paper_strict_builds_fixed_head_count() -> None:
    x, _ = _make_dataset(20)
    est = MHTSKClassifier(paper_strict=True, random_state=0)
    est._build_input_mfs(x)

    assert len(est._mhtsk_rules) == 600
    assert est._mhtsk_rule_feature_mask.shape[0] == 600


def test_mhtsk_paper_strict_trainer_updates_only_consequents() -> None:
    input_mfs = {
        "x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0), GaussianMF(mean=2.0, sigma=1.0)],
        "x2": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0), GaussianMF(mean=2.0, sigma=1.0)],
    }
    rules = [(1, 0), (0, 2), (2, 0)]
    rule_feature_mask = torch.tensor([[True, False], [False, True], [True, False]], dtype=torch.bool)
    model = MHTSKClassifierModel(input_mfs, rule_feature_mask, rules, n_classes=2, consequent_batch_norm=True)
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)

    ante_before = next(model.membership_layer.parameters()).detach().clone()
    cons_before = next(model.consequent_layer.parameters()).detach().clone()

    trainer = _MHTSKPaperStrictTrainer(epochs=1, learning_rate=1e-2, batch_size=8)
    trainer.fit(model, x, y)

    ante_after = next(model.membership_layer.parameters()).detach()
    cons_after = next(model.consequent_layer.parameters()).detach()

    assert torch.allclose(ante_before, ante_after)
    assert not torch.allclose(cons_before, cons_after)


def test_mhtsk_paper_strict_classifier_fit_retrains_with_strict_trainer() -> None:
    x, y = _make_dataset(20)
    est = MHTSKClassifier(
        paper_strict=True,
        consequent_batch_norm=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )

    est.fit(x, y)

    assert isinstance(est._get_trainer(), _MHTSKPaperStrictTrainer)
    assert est._extracted_rule_indices_ is not None


def test_mhtsk_paper_strict_regressor_fit_retrains_with_strict_trainer() -> None:
    x, y = _make_dataset(20)
    est = MHTSKRegressor(
        paper_strict=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )

    est.fit(x, y.astype(np.float32))

    assert isinstance(est._get_trainer(), _MHTSKPaperStrictTrainer)
    assert est._extracted_rule_indices_ is not None


def test_feature_coverage_rate() -> None:
    assert feature_coverage_rate(10, 2, 3) == pytest.approx(1.0 - (8.0 / 10.0) ** 3)


def test_feature_coverage_rate_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_features must be > 0"):
        feature_coverage_rate(0, 1, 1)
    with pytest.raises(ValueError, match="head_size must be between 1 and n_features"):
        feature_coverage_rate(3, 0, 1)
    with pytest.raises(ValueError, match="head_size must be between 1 and n_features"):
        feature_coverage_rate(3, 4, 1)
    with pytest.raises(ValueError, match="n_heads must be >= 0"):
        feature_coverage_rate(3, 1, -1)


def test_build_mhtsk_input_mfs_validates_inputs() -> None:
    x = np.random.rand(10, 3).astype(np.float32)
    feature_names = ["x1", "x2", "x3"]

    with pytest.raises(ValueError, match="head_size must be between 1 and the number of features"):
        _build_mhtsk_input_mfs(
            x,
            feature_names,
            n_heads=1,
            head_size=4,
            n_clusters=2,
            fcm_m=2.0,
            rule_sigma=1.0,
            instance_sample_fraction=1.0,
            random_state=0,
        )

    with pytest.raises(ValueError, match="n_heads must be > 0"):
        _build_mhtsk_input_mfs(
            x,
            feature_names,
            n_heads=0,
            head_size=1,
            n_clusters=2,
            fcm_m=2.0,
            rule_sigma=1.0,
            instance_sample_fraction=1.0,
            random_state=0,
        )

    with pytest.raises(ValueError, match=r"instance_sample_fraction must be in \(0, 1\]"):
        _build_mhtsk_input_mfs(
            x,
            feature_names,
            n_heads=1,
            head_size=1,
            n_clusters=2,
            fcm_m=2.0,
            rule_sigma=1.0,
            instance_sample_fraction=1.5,
            random_state=0,
        )


def test_build_mhtsk_input_mfs_full_instance_fraction_uses_full_data() -> None:
    x = np.random.rand(10, 3).astype(np.float32)
    feature_names = ["x1", "x2", "x3"]
    _, rules, rule_feature_mask = _build_mhtsk_input_mfs(
        x,
        feature_names,
        n_heads=1,
        head_size=1,
        n_clusters=2,
        fcm_m=2.0,
        rule_sigma=1.0,
        instance_sample_fraction=1.0,
        random_state=0,
    )

    assert len(rules) == 2
    assert rule_feature_mask.shape == (2, 3)


def test_resolve_mhtsk_scale_parameters_uses_paper_defaults() -> None:
    head_size, n_heads = _resolve_mhtsk_scale_parameters(
        n_features=1000,
        head_size=None,
        head_size_ratio=None,
        n_heads=None,
        fcr_target=0.85,
        h_value=None,
        sigma=1.0,
        xi=743.0,
    )

    assert head_size == 20
    assert n_heads == math.ceil(-math.log(1.0 - 0.85) * 1000 / head_size)


def test_resolve_mhtsk_scale_parameters_with_head_size_ratio() -> None:
    head_size, n_heads = _resolve_mhtsk_scale_parameters(
        n_features=1000,
        head_size=None,
        head_size_ratio=0.05,
        n_heads=None,
        fcr_target=None,
        h_value=3.0,
        sigma=1.0,
        xi=743.0,
    )

    assert head_size == 50
    assert n_heads == math.ceil(3.0 * 1000 / head_size)


def test_resolve_mhtsk_scale_parameters_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_features must be > 0"):
        _resolve_mhtsk_scale_parameters(
            n_features=0,
            head_size=None,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=0.85,
            h_value=None,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match="head_size must be between 1 and the number of features"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=11,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=0.85,
            h_value=None,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match="n_heads must be > 0"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=1,
            head_size_ratio=None,
            n_heads=0,
            fcr_target=0.85,
            h_value=None,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match=r"head_size_ratio must be in \(0, 1\]"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=None,
            head_size_ratio=1.5,
            n_heads=None,
            fcr_target=0.85,
            h_value=None,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match=r"fcr_target must be in \(0, 1\)"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=None,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=1.0,
            h_value=None,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match="h_value must be > 0"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=None,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=None,
            h_value=0.0,
            sigma=1.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match="sigma must be > 0"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=None,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=0.85,
            h_value=None,
            sigma=0.0,
            xi=743.0,
        )

    with pytest.raises(ValueError, match="xi must be > 0"):
        _resolve_mhtsk_scale_parameters(
            n_features=10,
            head_size=None,
            head_size_ratio=None,
            n_heads=None,
            fcr_target=0.85,
            h_value=None,
            sigma=1.0,
            xi=0.0,
        )


def test_mhtsk_regressor_estimator_samples_instances() -> None:
    x, y = _make_dataset(40)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        instance_sample_fraction=0.5,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y.astype(np.float32))
    assert est.model_.n_rules > 0


def test_extract_mhtsk_rule_indices_supervised() -> None:
    norm_w = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    selected = _extract_mhtsk_rule_indices(norm_w, y, 0.5, 0.5)
    assert len(selected) > 0


def test_extract_mhtsk_rule_indices_empty_rules() -> None:
    norm_w = torch.empty((0, 0), dtype=torch.float32)
    selected = _extract_mhtsk_rule_indices(norm_w, None, 0.5, 0.5)
    assert selected == []


def test_extract_mhtsk_rule_indices_fallback_when_empty() -> None:
    norm_w = torch.tensor([[0.1, 0.2], [0.1, 0.2]], dtype=torch.float32)
    y = torch.tensor([0, 0], dtype=torch.long)
    selected = _extract_mhtsk_rule_indices(norm_w, y, 0.0, 0.0)
    assert len(selected) == 1
    assert selected[0] in {0, 1}


def test_mhtsk_classifier_estimator_rule_extraction_with_validation_data() -> None:
    x, y = _make_dataset(40)
    x_val, y_val = _make_dataset(10)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        crcr_s=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y, x_val=x_val, y_val=y_val)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_regressor_estimator_rule_extraction_with_validation_data() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    x_val, y_val = _make_dataset(10)
    y_val = y_val.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y, x_val=x_val, y_val=y_val)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_classifier_estimator_rule_extraction_without_validation_data() -> None:
    x, y = _make_dataset(40)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        crcr_s=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_regressor_estimator_rule_extraction_without_validation_data() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        retrain_after_extraction=True,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_classifier_estimator_rule_extraction_without_retraining() -> None:
    x, y = _make_dataset(40)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        crcr_s=0.5,
        retrain_after_extraction=False,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_regressor_estimator_rule_extraction_without_retraining() -> None:
    x, y = _make_dataset(40)
    y = y.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        rule_extraction=True,
        crcr_us=0.5,
        retrain_after_extraction=False,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    assert est._extracted_rule_indices_ is not None
    assert len(est._extracted_rule_indices_) > 0


def test_mhtsk_classifier_extracted_model_rejects_empty_rule_list() -> None:
    x, y = _make_dataset(20)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    with pytest.raises(ValueError, match="At least one rule must be selected"):
        est._build_extracted_model(est.model_.input_mfs, [])


def test_mhtsk_regressor_extracted_model_rejects_empty_rule_list() -> None:
    x, y = _make_dataset(20)
    y = y.astype(np.float32)
    est = MHTSKRegressor(
        n_mfs=2,
        n_heads=2,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    est.fit(x, y)
    with pytest.raises(ValueError, match="At least one rule must be selected"):
        est._build_extracted_model(est.model_.input_mfs, [])


def test_mhtsk_input_builder_rejects_invalid_head_size() -> None:
    x, _ = _make_dataset(10)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=1,
        head_size=0,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    with pytest.raises(ValueError, match="head_size must be between 1 and the number of features"):
        est._build_input_mfs(x)


def test_mhtsk_input_builder_rejects_invalid_n_heads() -> None:
    x, _ = _make_dataset(10)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=0,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    with pytest.raises(ValueError, match="n_heads must be > 0"):
        est._build_input_mfs(x)


def test_mhtsk_input_builder_raises_when_fcm_fails(monkeypatch) -> None:
    x, _ = _make_dataset(10)

    class DummyFuzzyCMeans:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self, x: np.ndarray) -> None:
            self.cluster_centers_ = None

    monkeypatch.setattr("highfis.estimators._base.FuzzyCMeans", DummyFuzzyCMeans)

    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=1,
        head_size=1,
        fcm_m=2.0,
        rule_sigma=1.0,
        epochs=1,
        batch_size=16,
        random_state=0,
    )
    with pytest.raises(RuntimeError, match="FuzzyCMeans did not converge to a valid solution"):
        est._build_input_mfs(x)


def test_estimator_inspection_methods_for_mhtsk_classifier() -> None:
    x, y = _make_dataset(40)
    est = MHTSKClassifier(
        n_mfs=2,
        n_heads=2,
        head_size=2,
        epochs=3,
        learning_rate=1e-2,
        random_state=7,
        batch_size=16,
        verbose=False,
    )
    est.fit(x, y)

    info = est.inspect()
    assert info["n_rules"] == est.model_.n_rules
    assert len(info["rule_table"]) == est.model_.n_rules

    activations = est.rule_activation(x[:5])
    assert activations.shape == (5, est.model_.n_rules)
    assert np.allclose(np.sum(activations, axis=1), 1.0, atol=1e-5)

    importance = est.feature_importance()
    assert importance is not None
    assert importance.shape == (x.shape[1],)
    assert np.all(importance >= 0.0)
    assert np.isclose(np.sum(importance), 1.0)
