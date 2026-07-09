from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from highfis.memberships import GaussianMF
from highfis.models import (
    DGTSKClassifierModel,
    FSREADATSKClassifierModel,
    FSREADATSKRegressorModel,
)
from highfis.optim import (
    DGTrainer,
    FSRETrainer,
    GradientTrainer,
)
from highfis.optim._utils import (
    _get_optimizer_config,
    _log,
    _resolve_verbose,
)


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)] for i in range(n_inputs)}


# ==============================================================================
# Tests for highfis.optim._utils
# ==============================================================================


def test_resolve_verbose_values() -> None:
    assert _resolve_verbose(True) == 1
    assert _resolve_verbose(False) == 0
    assert _resolve_verbose(0) == 0
    assert _resolve_verbose(1) == 1
    assert _resolve_verbose(2) == 2
    assert _resolve_verbose(3) == 3

    with pytest.raises(TypeError, match="verbose must be an int"):
        _resolve_verbose("invalid")  # type: ignore

    with pytest.raises(ValueError, match="verbose must be between 0 and 3"):
        _resolve_verbose(-1)

    with pytest.raises(ValueError, match="verbose must be between 0 and 3"):
        _resolve_verbose(4)


def test_log_verbose(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    with caplog.at_level(logging.INFO):
        # Verbose = 0: should not log
        _log(logger, "message 0", verbose=0, min_level=2)
        assert "message 0" not in caplog.text

        # Verbose = 2, min_level = 2: should log
        _log(logger, "message 2", verbose=2, min_level=2)
        assert "message 2" in caplog.text


def test_get_optimizer_config_with_consequent_bn() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2, consequent_batch_norm=True)
    assert model.consequent_bn is not None

    _opt_class, param_groups = _get_optimizer_config(model, learning_rate=0.01, weight_decay=0.001)

    # Verify consequent_bn parameters are included in the param groups
    bn_params = list(model.consequent_bn.parameters())
    assert len(bn_params) > 0

    found_bn = False
    for group in param_groups:
        params_list = list(group["params"])
        if any(any(p is bp for bp in bn_params) for p in params_list):
            found_bn = True
            break
    assert found_bn, "consequent_bn parameters not found in optimizer config param groups"


def test_get_optimizer_config_optimizer_type_routing() -> None:
    """`optimizer_type` routes to the right optimizer; invalid values raise.

    Guards the DG-ALETSK fix: the paper uses plain Adam, but ``"adam"`` used to
    fall through silently to AdamW. Now ``"adam"`` maps to ``torch.optim.Adam``
    and unknown values raise instead of silently becoming AdamW.
    """
    model = DGTSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)

    model._optimizer_type = "sgd"
    assert _get_optimizer_config(model, 0.01, 0.0)[0] is torch.optim.SGD
    model._optimizer_type = "adam"
    assert _get_optimizer_config(model, 0.01, 0.0)[0] is torch.optim.Adam
    model._optimizer_type = "adamw"
    assert _get_optimizer_config(model, 0.01, 0.0)[0] is torch.optim.AdamW

    model._optimizer_type = "bogus"
    with pytest.raises(ValueError, match="unsupported optimizer_type"):
        _get_optimizer_config(model, 0.01, 0.0)


# ==============================================================================
# Tests for highfis.optim._gradient (GradientTrainer)
# ==============================================================================


def test_gradient_trainer_validation_errors() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    trainer = GradientTrainer(epochs=1)

    x = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,))

    # Wrong shape x
    with pytest.raises(ValueError, match="expected x shape"):
        trainer.fit(model, torch.randn(10, 2), y)

    # Wrong shape y
    with pytest.raises(ValueError, match="expected y shape"):
        trainer.fit(model, x, torch.randn(10, 2))

    # Negative ur_weight
    invalid_trainer = GradientTrainer(epochs=1, ur_weight=-0.5)
    with pytest.raises(ValueError, match="ur_weight must be >= 0"):
        invalid_trainer.fit(model, x, y)

    # Out of range ur_target
    invalid_trainer = GradientTrainer(epochs=1, ur_target=1.5)
    with pytest.raises(ValueError, match="ur_target must be in"):
        invalid_trainer.fit(model, x, y)

    # Invalid batch size
    invalid_trainer = GradientTrainer(epochs=1, batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        invalid_trainer.fit(model, x, y)

    # Validation set shape mismatch x_val
    with pytest.raises(ValueError, match="expected x_val shape"):
        trainer.fit(model, x, y, x_val=torch.randn(5, 2), y_val=torch.randint(0, 2, (5,)))

    # Validation set shape mismatch y_val
    with pytest.raises(ValueError, match="expected y_val shape"):
        trainer.fit(model, x, y, x_val=torch.randn(5, 3), y_val=torch.randint(0, 2, (5, 2)))


def test_gradient_trainer_early_stopping_accuracy() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)

    # We specify accuracy which is a maximized metric
    trainer = GradientTrainer(epochs=10, learning_rate=0.01, patience=2, verbose=0)

    x = torch.randn(20, 3)
    y = torch.randint(0, 2, (20,))

    history = trainer.fit(model, x, y, x_val=x, y_val=y, metrics=["accuracy"])
    assert "val_accuracy" in history
    assert len(history["val_accuracy"]) <= 10


def test_gradient_trainer_early_stopping_loss_minimized() -> None:
    model = FSREADATSKRegressorModel(_build_input_mfs(3, 2))

    # We specify mse which is a minimized metric, and verbose=2 to test early stopping logging
    trainer = GradientTrainer(epochs=10, learning_rate=0.01, patience=1, verbose=2)

    x = torch.randn(20, 3)
    y = torch.randn(20)

    history = trainer.fit(model, x, y, x_val=x, y_val=y, metrics=["mse"])
    assert "val_mse" in history
    assert "stopped_epoch" in history


def test_gradient_trainer_verbose_epoch_logs() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    x = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,))

    # Mock logger to intercept log messages
    logged = []

    def mock_log(level, msg, *args, **kwargs):
        logged.append(msg % args if args else msg)

    model.logger.log = mock_log  # type: ignore

    # Test verbose=1 (tqdm)
    trainer = GradientTrainer(epochs=2, verbose=1)
    trainer.fit(model, x, y)
    trainer.fit(model, x, y, x_val=x, y_val=y)

    # Test verbose=2 (logging with val)
    trainer = GradientTrainer(epochs=2, verbose=2)
    trainer.fit(model, x, y, x_val=x, y_val=y)
    assert any("epoch=" in msg for msg in logged)

    # Test verbose=3 (logging without val)
    logged.clear()
    trainer = GradientTrainer(epochs=2, verbose=3)
    trainer.fit(model, x, y)
    assert any("epoch=" in msg for msg in logged)


def test_gradient_trainer_classification_mseloss() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    x = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,))

    trainer = GradientTrainer(epochs=2, loss=nn.MSELoss())
    history = trainer.fit(model, x, y)
    assert len(history["train"]) == 2


# ==============================================================================
# Tests for highfis.optim._dg (DGTrainer)
# ==============================================================================


def test_dg_trainer_type_error() -> None:
    # Pass a non-DG model to DGTrainer
    model = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    trainer = DGTrainer(dg_epochs=1, finetune_epochs=1)

    x = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,))

    with pytest.raises(TypeError, match="model must implement DGModelProtocol"):
        trainer.fit(model, x, y)


def test_dg_trainer_first_order_conversion() -> None:
    # Fresh DGTSKClassifierModel starts with zero-order consequent.
    model = DGTSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    assert isinstance(model.consequent_layer, nn.Module)

    # We call DGTrainer directly.
    trainer = DGTrainer(dg_epochs=1, finetune_epochs=1, use_lse=False)
    x = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,))

    # Run fit to trigger convert_to_first_order() on line 243.
    trainer.fit(model, x, y)

    from highfis.layers import GatedClassificationConsequentLayer

    assert isinstance(model.consequent_layer, GatedClassificationConsequentLayer)


def test_dg_trainer_lambda_gates_not_tensor() -> None:
    class MockDGModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.membership_layer = nn.Linear(3, 3)
            self.rule_layer = nn.Module()
            self.consequent_layer = nn.Linear(3, 1)
            self.consequent_bn = None
            self._optimizer_type = "adamw"
            self.logger = logging.getLogger("mock_dg")
            self.task_type = "regression"
            self.n_inputs = 3

        def search_thresholds(
            self,
            x: Tensor,
            y: Tensor,
            zeta_lambda: Sequence[float] | None = None,
            zeta_theta: Sequence[float] | None = None,
            x_val: Tensor | None = None,
            y_val: Tensor | None = None,
            use_lse: bool = True,
            inplace: bool = True,
            verbose: bool = False,
            structural: bool = True,
        ) -> dict[str, Any]:
            return {"surviving_feature_indices": [0, 1, 2], "tau_lambda": 0.0, "tau_theta": 0.0}

        def default_criterion(self):
            return nn.MSELoss()

        def forward(self, x):
            return x.mean(dim=-1, keepdim=True)

        def _forward_train(self, x):
            return self(x), torch.ones(x.shape[0], 2)

    model = MockDGModel()
    trainer = DGTrainer(dg_epochs=0, finetune_epochs=0, use_lse=False)
    x = torch.randn(10, 3)
    y = torch.randn(10)

    # Verify fit runs and does not fail when lambda_gates is not a Tensor
    trainer.fit(model, x, y)  # type: ignore


# ==============================================================================
# Tests for highfis.optim._fsre (FSRETrainer)
# ==============================================================================


def test_fsre_trainer_type_error() -> None:
    # Pass a non-FSRE model to FSRETrainer
    model = DGTSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    trainer = FSRETrainer(fs_epochs=1, re_epochs=1, finetune_epochs=1)

    x = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,))

    with pytest.raises(TypeError, match="model must implement FSREModelProtocol"):
        trainer.fit(model, x, y)


def test_fsre_trainer_empty_features_fallback() -> None:
    # Set zeta_lambda to 0.0. The threshold tau = max_val.
    # Since no gate value can be strictly greater than max_val,
    # the list of surviving features will be empty, triggering the argmax fallback.
    model = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    trainer = FSRETrainer(
        fs_epochs=1,
        re_epochs=1,
        finetune_epochs=1,
        zeta_lambda=0.0,
        structural_pruning=True,
    )

    x = torch.randn(20, 3)
    y = torch.randint(0, 2, (20,))

    history = trainer.fit(model, x, y)
    # The history contains surviving feature indices, check that we kept exactly 1.
    assert len(history["surviving_feature_indices"]) == 1


def test_fsre_trainer_no_structural_pruning() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    trainer = FSRETrainer(
        fs_epochs=1,
        re_epochs=1,
        finetune_epochs=1,
        structural_pruning=False,
    )

    x = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,))

    _history = trainer.fit(model, x, y)
    # Pruning is disabled, so we expect the model's n_inputs to remain 3.
    assert model.n_inputs == 3


def test_gradient_trainer_custom_optimizer() -> None:
    model = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    x = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,))

    custom_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = GradientTrainer(epochs=1)
    # Fit using custom optimizer to cover the optimizer is not None branch (line 266)
    history = trainer.fit(model, x, y, optimizer=custom_opt)
    assert len(history["train"]) == 1


def test_predict_numpy_different_shapes() -> None:
    trainer = GradientTrainer(epochs=1)

    # 1D regression output
    class DummyModel1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.task_type = "regression"

        def eval(self):
            pass

        def train(self, mode):  # type: ignore
            pass

        def __call__(self, x):
            return torch.randn(x.shape[0])

    model = DummyModel1D()
    res = trainer._predict_numpy(model, torch.randn(5, 3))  # type: ignore
    assert res.ndim == 1

    # 2D multi-target regression output (shape[1] != 1)
    class DummyModel2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.task_type = "regression"

        def eval(self):
            pass

        def train(self, mode):  # type: ignore
            pass

        def __call__(self, x):
            return torch.randn(x.shape[0], 2)

    model = DummyModel2D()
    res = trainer._predict_numpy(model, torch.randn(5, 3))  # type: ignore
    assert res.shape == (5, 2)


def test_gradient_trainer_default_history_metrics() -> None:
    from highfis.models._fsre import FSREADATSKClassifierModel, FSREADATSKRegressorModel

    # 1. Classification
    model_cls = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    x = torch.randn(15, 3)
    y = torch.randint(0, 2, (15,))

    trainer = GradientTrainer(epochs=2)
    history_cls = trainer.fit(model_cls, x, y, x_val=x, y_val=y)

    assert "train_accuracy" in history_cls
    assert "val_accuracy" in history_cls
    assert len(history_cls["train_accuracy"]) == 2
    assert len(history_cls["val_accuracy"]) == 2

    # 2. Regression
    model_reg = FSREADATSKRegressorModel(_build_input_mfs(3, 2))
    y_reg = torch.randn(15)
    history_reg = trainer.fit(model_reg, x, y_reg, x_val=x, y_val=y_reg)

    assert "train_mse" in history_reg
    assert "val_mse" in history_reg
    assert len(history_reg["train_mse"]) == 2
    assert len(history_reg["val_mse"]) == 2


def test_gradient_trainer_scheduler_and_edge_cases() -> None:
    from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

    from highfis.models._fsre import FSREADATSKClassifierModel

    model = FSREADATSKClassifierModel(_build_input_mfs(3, 2), n_classes=2)
    x = torch.randn(15, 3)
    y = torch.randint(0, 2, (15,))

    # 1. StepLR scheduler
    trainer = GradientTrainer(epochs=2, learning_rate=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    sched = StepLR(opt, step_size=1, gamma=0.5)
    history = trainer.fit(model, x, y, optimizer=opt, scheduler=sched)
    assert len(history["lr"]) == 2
    assert history["lr"][1] == 0.025

    # 2. ReduceLROnPlateau scheduler (with validation)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    sched_plateau = ReduceLROnPlateau(opt, factor=0.1)
    history2 = trainer.fit(model, x, y, x_val=x, y_val=y, optimizer=opt, scheduler=sched_plateau)
    assert len(history2["lr"]) == 2

    # 3. ReduceLROnPlateau scheduler (no validation)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    sched_plateau_no_val = ReduceLROnPlateau(opt, factor=0.1)
    history3 = trainer.fit(model, x, y, optimizer=opt, scheduler=sched_plateau_no_val)
    assert len(history3["lr"]) == 2

    # 4. metrics=[] (empty list) with validation, and early stopping with verbose=2
    trainer_es = GradientTrainer(epochs=5, patience=1, verbose=2)
    history_es = trainer_es.fit(model, x, y, x_val=x, y_val=y, metrics=[])
    assert history_es["stopped_epoch"] < 5

    # 5. classification with validation but accuracy is not in metrics (e.g. metrics=["precision_macro"])
    history_cls = trainer.fit(model, x, y, x_val=x, y_val=y, metrics=["precision_macro"])
    assert "val_accuracy" in history_cls
    assert "val_acc" in history_cls
