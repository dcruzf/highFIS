from __future__ import annotations

import pytest
import torch
from torch import nn

from highfis.gates import (
    BaseGate,
    ExpGate,
    InvExpGate,
    MGate,
    SigmoidGate,
    SignedExpGate,
    gate1,
    gate2,
    gate3,
    gate4,
    gate_m,
    resolve_gate_fn,
)


def test_base_gate_raises_not_implemented() -> None:
    gate = BaseGate()
    with pytest.raises(NotImplementedError):
        gate(torch.randn(3))


def test_base_gate_default_init_params() -> None:
    class MinimalGate(BaseGate):
        def forward(self, u: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(u)

    gate = MinimalGate()
    param = nn.Parameter(torch.zeros(8))
    gate.init_params_(param)
    data = param.detach()
    assert data.min() >= 0.01
    assert data.max() <= 0.1


def test_sigmoid_gate() -> None:
    gate = SigmoidGate()
    assert gate.is_nonneg is True

    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(gate(x), torch.sigmoid(x))

    param = nn.Parameter(torch.zeros(10))
    gate.init_params_(param)
    data = param.detach()
    assert data.min() >= -5.5
    assert data.max() <= -4.5


def test_exp_gate() -> None:
    gate = ExpGate(k=2.0)
    assert gate.is_nonneg is True
    assert gate.k == 2.0

    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(gate(x), 1.0 - torch.exp(-2.0 * x.pow(2)))

    param = nn.Parameter(torch.zeros(10))
    gate.init_params_(param)
    data = param.detach()
    assert data.min() >= 0.001
    assert data.max() <= 0.01


def test_inv_exp_gate() -> None:
    gate = InvExpGate()
    assert gate.is_nonneg is True

    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(gate(x), torch.exp(-x.pow(2)))

    param = nn.Parameter(torch.zeros(100))
    gate.init_params_(param)
    data = param.detach()
    # Check mean/std roughly matches normal(3.0, 0.2)
    assert 2.0 < data.mean().item() < 4.0


def test_signed_exp_gate() -> None:
    gate = SignedExpGate()
    assert gate.is_nonneg is False

    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(gate(x), x * torch.sqrt(torch.exp(1.0 - x.pow(2))))

    param = nn.Parameter(torch.zeros(10))
    gate.init_params_(param)
    data = param.detach()
    assert data.min() >= 0.005
    assert data.max() <= 0.015


def test_m_gate() -> None:
    gate = MGate()
    assert gate.is_nonneg is True

    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(gate(x), x.pow(2) * torch.exp(1.0 - x.pow(2)))

    param = nn.Parameter(torch.zeros(10))
    gate.init_params_(param)
    data = param.detach()
    assert data.min() >= 0.01
    assert data.max() <= 0.1


def test_gate_singletons() -> None:
    assert isinstance(gate1, SigmoidGate)
    assert isinstance(gate2, ExpGate)
    assert gate2.k == 1.0
    assert isinstance(gate3, InvExpGate)
    assert isinstance(gate4, SignedExpGate)
    assert isinstance(gate_m, MGate)


def test_resolve_gate_fn() -> None:
    assert resolve_gate_fn("gate1") is gate1
    assert resolve_gate_fn("gate2") is gate2
    assert resolve_gate_fn("gate3") is gate3
    assert resolve_gate_fn("gate4") is gate4
    assert resolve_gate_fn("gate_m") is gate_m

    default = resolve_gate_fn(None)
    assert isinstance(default, ExpGate)
    assert default.k == 10.0

    def custom(u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u)

    assert resolve_gate_fn(custom) is custom

    with pytest.raises(ValueError, match="unsupported gate function"):
        resolve_gate_fn("invalid_gate")
