from __future__ import annotations

from dataclasses import replace

import torch

from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.protocol import Run
from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_18_membrane_history as exp


def _p():
    return replace(
        smoke_protocol(),
        width=12,
        input_channels=3,
        total_channels=3,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=4,
        max_epochs=2,
        min_epochs=1,
        patience=1,
    )


def test_axes_are_locked():
    assert exp.SEEDS == (11, 23, 37)
    assert exp.CASES == ("U_NORMAL", "U_DETACH")
    assert exp.SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp.SUFFIX_STEPS == 16
    assert len(exp.specs()) == 6


def test_tau_groups_are_dimension_matched():
    values = exp._slow_values((2, 3, 4), 128)
    assert values.shape == (128,)
    assert torch.all(values[:43] == 0.75)
    assert torch.all(values[43:86] == 0.875)
    assert torch.all(values[86:] == 0.9375)


def test_normal_and_detach_forward_are_identical():
    p = _p()
    normal = exp.UHistoryNet(exp.ExpSpec("U_NORMAL", 11), p)
    detach = exp.UHistoryNet(exp.ExpSpec("U_DETACH", 11), p)
    detach.load_state_dict(normal.state_dict(), strict=True)
    x = torch.randn(4, p.steps, p.input_channels)
    lengths = torch.tensor([16, 15, 12, 9])
    a, b = normal(x, lengths), detach(x, lengths)
    for key in ("evidence",):
        assert torch.equal(a[key], b[key])
    for key in ("spike", "pre_reset", "post_reset", "synaptic"):
        for av, bv in zip(a[key], b[key]):
            assert torch.equal(av, bv)


def test_normal_and_detach_backward_differ_after_reset():
    p = _p()
    normal = exp.UHistoryNet(exp.ExpSpec("U_NORMAL", 11), p)
    detach = exp.UHistoryNet(exp.ExpSpec("U_DETACH", 11), p)
    detach.load_state_dict(normal.state_dict(), strict=True)
    with torch.no_grad():
        for model in (normal, detach):
            for layer in model.layers:
                layer.weight.fill_(0.8)
            model.head.weight.fill_(0.3)
    x = torch.ones(2, p.steps, p.input_channels)
    lengths = torch.full((2,), p.steps, dtype=torch.long)
    for model in (normal, detach):
        model.zero_grad(set_to_none=True)
        model(x, lengths)["evidence"].sum().backward()
    assert not torch.equal(normal.layers[0].weight.grad, detach.layers[0].weight.grad)


def test_parameter_initialization_matches_core_o0():
    p = _p()
    spec = exp.ExpSpec("U_NORMAL", 11)
    u = exp.UHistoryNet(spec, p)
    o0 = BenchmarkNet(Run("O0", 11, "01_objective", shifts=exp.SHIFTS, objective="wcce"), p)
    assert torch.equal(u.layers[0].weight, o0.layers[0].weight)
    assert torch.equal(u.layers[1].weight, o0.layers[1].weight)
    assert torch.equal(u.head.weight, o0.head.weight)


def test_subthreshold_pole_swap_equivalence():
    p = _p()
    o0 = BenchmarkNet(Run("O0", 11, "01_objective", shifts=exp.SHIFTS, objective="wcce"), p)
    u = exp.UHistoryNet(exp.ExpSpec("U_NORMAL", 11), p)
    with torch.no_grad():
        for layer in o0.layers:
            layer.weight.mul_(1e-3)
    u.layers.load_state_dict(o0.layers.state_dict())
    u.head.load_state_dict(o0.head.state_dict())
    x = 1e-3 * torch.randn(3, p.steps, p.input_channels)
    lengths = torch.tensor([16, 13, 10])
    a, b = o0(x, lengths), u(x, lengths)
    assert all(int(v.sum()) == 0 for v in a["spike"])
    assert all(int(v.sum()) == 0 for v in b["spike"])
    for av, bv in zip(a["pre_reset"], b["pre_reset"]):
        assert torch.allclose(av, bv, atol=2e-6, rtol=1e-6)


def test_suffix_logits_uses_only_last_valid_steps():
    evidence = torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1)
    lengths = torch.tensor([5])
    result = exp._suffix_logits(evidence, lengths, 2)
    assert torch.equal(result, torch.tensor([[4.5]]))
