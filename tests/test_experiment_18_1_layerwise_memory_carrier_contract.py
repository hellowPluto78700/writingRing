from __future__ import annotations

from dataclasses import replace

import torch

from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.protocol import Run
from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_18_1_layerwise_memory_carrier as exp
from scripts import experiment_18_membrane_history as exp18


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


def test_axes_and_layer_mapping_are_locked():
    assert exp.SEEDS == (11, 23, 37)
    assert exp.CASES == ("UI", "IU")
    assert exp.CARRIERS["UI"] == ("U", "I")
    assert exp.CARRIERS["IU"] == ("I", "U")
    assert exp.CARRIERS["II_REF"] == ("I", "I")
    assert exp.CARRIERS["UU_REF"] == ("U", "U")
    assert len(exp.specs()) == 6


def test_initialization_is_paired_across_all_four_corners_and_core():
    p = _p()
    models = [
        exp.HybridMemoryNet(exp.ExpSpec("UI", 11), p),
        exp.HybridMemoryNet(exp.ExpSpec("IU", 11), p),
        exp.HybridMemoryNet(exp.ExpSpec("UI", 11), p, carriers=("I", "I")),
        exp.HybridMemoryNet(exp.ExpSpec("UI", 11), p, carriers=("U", "U")),
    ]
    core = BenchmarkNet(Run("O0", 11, "01_objective", shifts=exp.SHIFTS, objective="wcce"), p)
    for model in models:
        assert torch.equal(model.layers[0].weight, core.layers[0].weight)
        assert torch.equal(model.layers[1].weight, core.layers[1].weight)
        assert torch.equal(model.head.weight, core.head.weight)


def test_ii_corner_reproduces_core_forward():
    p = _p()
    core = BenchmarkNet(Run("O0", 11, "01_objective", shifts=exp.SHIFTS, objective="wcce"), p)
    ii = exp.HybridMemoryNet(exp.ExpSpec("UI", 11), p, carriers=("I", "I"))
    ii.layers.load_state_dict(core.layers.state_dict())
    ii.head.load_state_dict(core.head.state_dict())
    x = torch.randn(4, p.steps, p.input_channels)
    lengths = torch.tensor([16, 15, 12, 9])
    a, b = core(x, lengths), ii(x, lengths)
    assert torch.equal(a["evidence"], b["evidence"])
    for key in ("spike", "pre_reset"):
        for av, bv in zip(a[key], b[key]):
            assert torch.equal(av, bv)


def test_uu_corner_reproduces_exp18_unormal_forward():
    p = _p()
    old = exp18.UHistoryNet(exp18.ExpSpec("U_NORMAL", 11), p)
    uu = exp.HybridMemoryNet(exp.ExpSpec("UI", 11), p, carriers=("U", "U"))
    uu.layers.load_state_dict(old.layers.state_dict())
    uu.head.load_state_dict(old.head.state_dict())
    x = torch.randn(4, p.steps, p.input_channels)
    lengths = torch.tensor([16, 14, 11, 8])
    a, b = old(x, lengths), uu(x, lengths)
    assert torch.equal(a["evidence"], b["evidence"])
    for key in ("spike", "pre_reset"):
        for av, bv in zip(a[key], b[key]):
            assert torch.equal(av, bv)


def test_ui_and_iu_are_distinct_after_spiking():
    p = _p()
    ui = exp.HybridMemoryNet(exp.ExpSpec("UI", 11), p)
    iu = exp.HybridMemoryNet(exp.ExpSpec("IU", 11), p)
    iu.load_state_dict(ui.state_dict(), strict=True)
    with torch.no_grad():
        for model in (ui, iu):
            for layer in model.layers:
                layer.weight.fill_(0.8)
            model.head.weight.fill_(0.3)
    x = torch.ones(2, p.steps, p.input_channels)
    lengths = torch.full((2,), p.steps, dtype=torch.long)
    a, b = ui(x, lengths), iu(x, lengths)
    assert int(a["spike"][0].sum()) > 0
    assert int(b["spike"][0].sum()) > 0
    # Binary spikes may saturate to the same all-one pattern under this
    # deliberately strong drive. The carrier intervention must instead be
    # visible in the post-spike state trajectory: resetting the slow U state
    # versus the fast U state changes subsequent pre-reset values.
    assert not torch.equal(a["pre_reset"][0], b["pre_reset"][0])


def test_subthreshold_single_layer_pole_swap_is_equivalent():
    p = _p()
    slow = exp._slow_values((2, 3, 4), p.width)
    fast = torch.exp(torch.tensor(-(1000.0 / p.fs) / p.tau_mem_ms))
    drive = 1e-4 * torch.randn(3, p.steps, p.width)

    def run(carrier: str):
        syn = torch.zeros(3, p.width)
        mem = torch.zeros(3, p.width)
        pre = []
        for t in range(p.steps):
            syn_decay, mem_decay = (slow, fast) if carrier == "I" else (fast, slow)
            syn = syn_decay * syn + drive[:, t]
            value = mem_decay * mem + syn
            assert bool((value.abs() < p.threshold).all())
            mem = value
            pre.append(value)
        return torch.stack(pre, 1)

    assert torch.allclose(run("I"), run("U"), atol=2e-6, rtol=1e-6)


def test_factorial_case_order_is_stable():
    assert ("II_REF", "UI", "IU", "UU_REF") == ("II_REF", *exp.CASES, "UU_REF")
