"""Tests for LogTSK models (Cui, Wu & Xu, IEEE Trans. Fuzzy Syst. 2021, §III-A).

LogTSK normalizes firing strengths using the scale-invariant inverse-log formula:

    f̄_r = (1/|Z_r|) / Σ_i (1/|Z_i|)

where Z_r = log f_r = Σ_d log μ_{r,d} ≤ 0.  The key property is scale-invariance:
multiplying all Z_r by the same positive constant k does not change the output,
making this defuzzifier immune to softmax saturation as input dimension D grows.
"""

from __future__ import annotations

import pytest
import torch

from highfis.defuzzifiers import InvLogDefuzzifier
from highfis.memberships import GaussianMF
from highfis.models import LogTSKClassifierModel, LogTSKRegressorModel


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


# =====================================================================
# LogTSKClassifierModel
# =====================================================================


class TestLogTSKClassifierInit:
    def test_rejects_empty_input_mfs(self) -> None:
        with pytest.raises(ValueError, match="input_mfs must not be empty"):
            LogTSKClassifierModel({}, n_classes=2)

    def test_rejects_invalid_n_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            LogTSKClassifierModel(_build_input_mfs(), n_classes=1)

    def test_default_defuzzifier_is_inv_log(self) -> None:
        model = LogTSKClassifierModel(_build_input_mfs(), n_classes=3)
        assert isinstance(model.defuzzifier, InvLogDefuzzifier)

    def test_custom_defuzzifier_is_accepted(self) -> None:
        from highfis.defuzzifiers import LogSumDefuzzifier

        custom = LogSumDefuzzifier(temperature=0.5)
        model = LogTSKClassifierModel(_build_input_mfs(), n_classes=3, defuzzifier=custom)
        assert model.defuzzifier is custom


class TestLogTSKClassifierForward:
    def test_forward_shape(self) -> None:
        model = LogTSKClassifierModel(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        logits = model.forward(x)
        assert logits.shape == (8, 3)

    def test_predict_proba_sums_to_one(self) -> None:
        model = LogTSKClassifierModel(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        proba = model.predict_proba(x)
        assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)

    def test_predict_returns_class_indices(self) -> None:
        model = LogTSKClassifierModel(_build_input_mfs(), n_classes=3)
        x = torch.randn(8, 3)
        pred = model.predict(x)
        assert pred.shape == (8,)

    def test_antecedent_norm_w_sums_to_one(self) -> None:
        """InvLogDefuzzifier output sums to 1."""
        model = LogTSKClassifierModel(_build_input_mfs(), n_classes=2)
        x = torch.randn(6, 3)
        norm_w = model.forward_antecedents(x)
        assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


class TestLogTSKClassifierMath:
    def test_inv_log_scale_invariance(self) -> None:
        """Multiplying all log-firing strengths by k > 0 must not change output.

        This is the key property that makes InvLogDefuzzifier immune to
        softmax saturation: if we scale Z_r → k·Z_r for all r, then
        (1/|k·Z_r|) / Σ(1/|k·Z_i|) = (1/|Z_r|) / Σ(1/|Z_i|).
        """
        defuzz = InvLogDefuzzifier()
        torch.manual_seed(0)
        # Values in [0.4, 0.9] — safely above eps even after cubing
        w = torch.rand(10, 4) * 0.5 + 0.4
        out1 = defuzz(w)
        # w^3 → Z_r scaled by 3: should give identical normalized output
        out2 = defuzz(w**3.0)
        assert torch.allclose(out1, out2, atol=1e-5), (
            "InvLogDefuzzifier is not scale-invariant: outputs differ after scaling Z_r by k"
        )

    def test_inv_log_not_saturation_prone(self) -> None:
        """Unlike softmax, InvLogDefuzzifier does not saturate with high D.

        Design rule 1 to fire at 0.9^D and rules 2-4 at 0.5^D.
        Softmax(log(w)) saturates (max ≈ 1); InvLog stays bounded.
        """
        from highfis.defuzzifiers import SoftmaxLogDefuzzifier

        N, R, D = 20, 4, 100
        w_high = torch.full((N, 1), 0.9) ** D
        w_low = torch.full((N, R - 1), 0.5) ** D
        w = torch.cat([w_high, w_low], dim=1)

        inv_log_defuzz = InvLogDefuzzifier()
        softmax_defuzz = SoftmaxLogDefuzzifier()

        norm_inv = inv_log_defuzz(w)
        norm_soft = softmax_defuzz(w)

        max_inv = norm_inv.max(dim=1).values.mean().item()
        max_soft = norm_soft.max(dim=1).values.mean().item()

        # Softmax should be very close to 1 (saturated); InvLog should not
        assert max_soft > 0.97, f"Expected softmax saturation, got {max_soft:.4f}"
        assert max_inv < 0.99, f"Expected InvLog not saturated, got {max_inv:.4f}"

    def test_inv_log_defuzzifier_values(self) -> None:
        """Verify numerics for a known small case."""
        defuzz = InvLogDefuzzifier()
        # Two rules: w1=e^{-1}, w2=e^{-2} → Z1=-1, Z2=-2
        # 1/|Z1|=1, 1/|Z2|=0.5 → normalized: [2/3, 1/3]
        w = torch.tensor([[torch.e**-1, torch.e**-2]])
        out = defuzz(w)
        expected = torch.tensor([[2.0 / 3.0, 1.0 / 3.0]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_high_dimensional_pipeline_no_underflow(self) -> None:
        """LogTSKClassifierModel must not collapse to uniform weights for D=784.

        Regression test for the prod-t-norm underflow bug: the product of 784
        Gaussian MF values underflows to 0.0 in float32, making ALL raw firing
        levels exactly zero (raw_zero_fraction == 1). The gmean t-norm avoids
        this by computing exp(mean(log(mu))) instead of product(mu).
        """
        torch.manual_seed(0)
        D, N, R = 784, 30, 5
        # Spread MF centers widely so rules have distinct firing levels
        input_mfs = {f"x{i}": [GaussianMF(mean=float(j - 2), sigma=0.5) for j in range(R)] for i in range(D)}
        model = LogTSKClassifierModel(input_mfs, n_classes=2, rule_base="coco")

        x = torch.randn(N, D)
        raw_w = model.rule_layer(model.membership_layer(x))  # (N, R), before defuzz

        # With gmean (the fix), raw firing levels must not all be exactly 0
        zero_frac = (raw_w == 0.0).float().mean().item()
        assert zero_frac == 0.0, (
            f"LogTSK raw firing levels are {zero_frac * 100:.1f}% zero — underflow not fixed (prod t-norm still used?)"
        )


class TestLogTSKClassifierFit:
    def test_fit_returns_history(self) -> None:
        torch.manual_seed(1)
        model = LogTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,), dtype=torch.long)
        history = model.fit(x, y, epochs=4, learning_rate=1e-2, batch_size=5)
        assert len(history["train"]) == 4

    def test_early_stopping_with_val_data(self) -> None:
        torch.manual_seed(42)
        model = LogTSKClassifierModel(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
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
# LogTSKRegressorModel
# =====================================================================


class TestLogTSKRegressorInit:
    def test_rejects_empty_input_mfs(self) -> None:
        with pytest.raises(ValueError, match="input_mfs must not be empty"):
            LogTSKRegressorModel({})

    def test_default_defuzzifier_is_inv_log(self) -> None:
        model = LogTSKRegressorModel(_build_input_mfs())
        assert isinstance(model.defuzzifier, InvLogDefuzzifier)

    def test_custom_defuzzifier_is_accepted(self) -> None:
        from highfis.defuzzifiers import LogSumDefuzzifier

        custom = LogSumDefuzzifier(temperature=0.5)
        model = LogTSKRegressorModel(_build_input_mfs(), defuzzifier=custom)
        assert model.defuzzifier is custom


class TestLogTSKRegressorForward:
    def test_forward_shape(self) -> None:
        model = LogTSKRegressorModel(_build_input_mfs())
        x = torch.randn(8, 3)
        out = model.forward(x)
        assert out.shape == (8, 1)

    def test_predict_returns_1d(self) -> None:
        model = LogTSKRegressorModel(_build_input_mfs())
        x = torch.randn(8, 3)
        pred = model.predict(x)
        assert pred.shape == (8,)

    def test_antecedent_norm_w_sums_to_one(self) -> None:
        model = LogTSKRegressorModel(_build_input_mfs())
        x = torch.randn(6, 3)
        norm_w = model.forward_antecedents(x)
        assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


class TestLogTSKRegressorFit:
    def test_fit_returns_history(self) -> None:
        torch.manual_seed(1)
        model = LogTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
        x = torch.randn(20, 2)
        y = torch.randn(20)
        history = model.fit(x, y, epochs=4, learning_rate=1e-2, batch_size=5)
        assert len(history["train"]) == 4

    def test_fit_loss_decreases(self) -> None:
        torch.manual_seed(42)
        model = LogTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
        x = torch.randn(40, 2)
        y = x[:, 0] + 0.5 * x[:, 1]
        history = model.fit(x, y, epochs=50, learning_rate=1e-2)
        assert history["train"][-1] < history["train"][0]

    def test_early_stopping_with_val_data(self) -> None:
        torch.manual_seed(42)
        model = LogTSKRegressorModel(_build_input_mfs(n_inputs=2, n_mfs=2))
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
