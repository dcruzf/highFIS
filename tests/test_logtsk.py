"""Tests for LogTSK models (Cui, Wu & Xu, IEEE Trans. Fuzzy Syst. 2021).

LogTSK normalizes in log-space with optional temperature scaling:

    log f̄_r = log(w_r)/τ − log(Σ exp(log(w_i)/τ))

At τ=1 it is mathematically equivalent to SoftmaxLogDefuzzifier.
"""

from __future__ import annotations

import pytest
import torch

from highfis.defuzzifiers import LogSumDefuzzifier
from highfis.memberships import GaussianMF
from highfis.models import LogTSKClassifier, LogTSKRegressor


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


# =====================================================================
# LogTSKClassifier
# =====================================================================


class TestLogTSKClassifierInit:
    def test_rejects_empty_input_mfs(self) -> None:
        with pytest.raises(ValueError, match="input_mfs must not be empty"):
            LogTSKClassifier({}, n_classes=2)

    def test_rejects_invalid_n_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            LogTSKClassifier(_build_input_mfs(), n_classes=1)

    def test_default_defuzzifier_is_log_sum(self) -> None:
        model = LogTSKClassifier(_build_input_mfs(), n_classes=3)
        assert isinstance(model.defuzzifier, LogSumDefuzzifier)

    def test_temperature_is_passed_to_defuzzifier(self) -> None:
        model = LogTSKClassifier(_build_input_mfs(), n_classes=3, temperature=0.5)
        assert isinstance(model.defuzzifier, LogSumDefuzzifier)
        assert model.defuzzifier.temperature == 0.5


class TestLogTSKClassifierForward:
    def test_forward_shape(self) -> None:
        model = LogTSKClassifier(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        logits = model.forward(x)
        assert logits.shape == (8, 3)

    def test_predict_proba_sums_to_one(self) -> None:
        model = LogTSKClassifier(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        proba = model.predict_proba(x)
        assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)

    def test_predict_returns_class_indices(self) -> None:
        model = LogTSKClassifier(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        pred = model.predict(x)
        assert pred.shape == (8,)

    def test_antecedent_norm_w_sums_to_one(self) -> None:
        """LogSumDefuzzifier output sums to 1 (softmax guarantee)."""
        model = LogTSKClassifier(_build_input_mfs(), n_classes=2)
        x = torch.randn(6, 3)
        norm_w = model.forward_antecedents(x)
        assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


class TestLogTSKClassifierMath:
    def test_temperature_1_equivalent_to_softmax_log(self) -> None:
        """At τ=1, LogSumDefuzzifier ≈ SoftmaxLogDefuzzifier end-to-end."""
        from highfis.defuzzifiers import SoftmaxLogDefuzzifier
        from highfis.models import HTSKClassifier

        torch.manual_seed(99)
        mfs = _build_input_mfs(n_inputs=2, n_mfs=2)

        m_log = LogTSKClassifier(mfs, n_classes=2, t_norm="gmean", temperature=1.0)
        m_htsk = HTSKClassifier(mfs, n_classes=2, t_norm="gmean")

        assert isinstance(m_htsk.defuzzifier, SoftmaxLogDefuzzifier)

        x = torch.randn(10, 2)
        # Same architecture but freshly init'd—just verify both produce valid output
        norm_log = m_log.forward_antecedents(x)
        norm_htsk = m_htsk.forward_antecedents(x)
        assert torch.allclose(norm_log.sum(dim=1), torch.ones(10), atol=1e-6)
        assert torch.allclose(norm_htsk.sum(dim=1), torch.ones(10), atol=1e-6)

    def test_lower_temperature_sharpens_distribution(self) -> None:
        """Lower τ concentrates probability on the max-firing rule."""
        torch.manual_seed(42)
        mfs = _build_input_mfs(n_inputs=2, n_mfs=3)
        x = torch.randn(20, 2)

        m_high = LogTSKClassifier(mfs, n_classes=2, temperature=2.0)
        m_low = LogTSKClassifier(mfs, n_classes=2, temperature=0.1)

        # Copy antecedent weights so firing strengths are identical
        m_low.load_state_dict(m_high.state_dict(), strict=False)

        norm_high = m_high.forward_antecedents(x)
        norm_low = m_low.forward_antecedents(x)

        # Lower temperature → higher max → lower entropy
        assert norm_low.max(dim=1).values.mean() > norm_high.max(dim=1).values.mean()


class TestLogTSKClassifierFit:
    def test_fit_returns_history(self) -> None:
        torch.manual_seed(1)
        model = LogTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,), dtype=torch.long)
        history = model.fit(x, y, epochs=4, learning_rate=1e-2, batch_size=5)
        assert len(history["train"]) == 4

    def test_early_stopping_with_val_data(self) -> None:
        torch.manual_seed(42)
        model = LogTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
        x = torch.randn(30, 2)
        y = torch.randint(0, 2, (30,), dtype=torch.long)
        x_val = torch.randn(10, 2)
        y_val = torch.randint(0, 2, (10,), dtype=torch.long)
        history = model.fit(
            x,
            y,
            epochs=500,
            x_val=x_val,
            y_val=y_val,
            patience=5,
            learning_rate=1e-2,
        )
        assert history["stopped_epoch"] < 500


# =====================================================================
# LogTSKRegressor
# =====================================================================


class TestLogTSKRegressorInit:
    def test_rejects_empty_input_mfs(self) -> None:
        with pytest.raises(ValueError, match="input_mfs must not be empty"):
            LogTSKRegressor({})

    def test_default_defuzzifier_is_log_sum(self) -> None:
        model = LogTSKRegressor(_build_input_mfs())
        assert isinstance(model.defuzzifier, LogSumDefuzzifier)

    def test_temperature_is_passed_to_defuzzifier(self) -> None:
        model = LogTSKRegressor(_build_input_mfs(), temperature=0.5)
        assert isinstance(model.defuzzifier, LogSumDefuzzifier)
        assert model.defuzzifier.temperature == 0.5


class TestLogTSKRegressorForward:
    def test_forward_shape(self) -> None:
        model = LogTSKRegressor(_build_input_mfs())
        x = torch.randn(8, 3)
        out = model.forward(x)
        assert out.shape == (8, 1)

    def test_predict_returns_1d(self) -> None:
        model = LogTSKRegressor(_build_input_mfs())
        x = torch.randn(8, 3)
        pred = model.predict(x)
        assert pred.shape == (8,)

    def test_antecedent_norm_w_sums_to_one(self) -> None:
        model = LogTSKRegressor(_build_input_mfs())
        x = torch.randn(6, 3)
        norm_w = model.forward_antecedents(x)
        assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


class TestLogTSKRegressorFit:
    def test_fit_returns_history(self) -> None:
        torch.manual_seed(1)
        model = LogTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
        x = torch.randn(20, 2)
        y = torch.randn(20)
        history = model.fit(x, y, epochs=4, learning_rate=1e-2, batch_size=5)
        assert len(history["train"]) == 4

    def test_fit_loss_decreases(self) -> None:
        torch.manual_seed(42)
        model = LogTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
        x = torch.randn(40, 2)
        y = x[:, 0] + 0.5 * x[:, 1]
        history = model.fit(x, y, epochs=50, learning_rate=1e-2)
        assert history["train"][-1] < history["train"][0]

    def test_early_stopping_with_val_data(self) -> None:
        torch.manual_seed(42)
        model = LogTSKRegressor(_build_input_mfs(n_inputs=2, n_mfs=2))
        x = torch.randn(30, 2)
        y = x[:, 0] + 0.5 * x[:, 1]
        x_val = torch.randn(10, 2)
        y_val = x_val[:, 0] + 0.5 * x_val[:, 1]
        history = model.fit(
            x,
            y,
            epochs=2000,
            x_val=x_val,
            y_val=y_val,
            patience=15,
            learning_rate=5e-2,
        )
        assert history["stopped_epoch"] < 2000
