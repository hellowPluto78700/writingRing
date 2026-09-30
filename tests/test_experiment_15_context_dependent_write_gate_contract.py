from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_15_context_dependent_write_gate as exp


def _small_protocol():
    return replace(
        smoke_protocol(),
        width=6,
        input_channels=3,
        total_channels=3,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=4,
        max_epochs=2,
        min_epochs=1,
        patience=1,
        relative_bins=4,
        shuffle_seeds=(101,),
    )


def test_manifest_and_parallel_task_counts() -> None:
    assert exp.SEEDS == (11, 23, 37)
    assert exp.CASES == ("C0", "GF", "GJ")
    assert len(exp.phase1_specs()) == 9
    assert len(exp.phase1_5_specs()) == 6
    assert {spec.case for spec in exp.phase1_5_specs()} == {"GF", "GJ"}


def test_gate_initialization_is_function_preserving() -> None:
    p = _small_protocol()
    run = exp._exp_run(exp.ExpSpec("GJ", 11))
    baseline = BenchmarkNet(run, p)
    gated = exp.ContextWriteGateNet(run, p)
    gated.load_baseline_state(baseline.state_dict())

    generator = torch.Generator().manual_seed(123)
    x = torch.rand(4, p.steps, p.input_channels, generator=generator)
    lengths = torch.tensor([16, 14, 11, 9], dtype=torch.long)

    with torch.no_grad():
        base = baseline(x, lengths)
        gate = gated(x, lengths)

    assert torch.allclose(gate["gate_m"][:, :9], torch.ones_like(gate["gate_m"][:, :9]))
    assert torch.allclose(base["evidence"], gate["evidence"], atol=1e-6, rtol=0)
    assert torch.equal(base["spike"][0], gate["spike"][0])
    assert torch.equal(base["spike"][1], gate["spike"][1])


def test_gf_freezes_all_non_gate_parameters() -> None:
    p = _small_protocol()
    model = exp.ContextWriteGateNet(exp._exp_run(exp.ExpSpec("GF", 11)), p)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("gate_"))
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"gate_input.weight", "gate_history.weight", "gate_bias"}


def test_a2_preserves_sequence_mean_and_removes_within_sequence_variation() -> None:
    p = _small_protocol()
    model = exp.ContextWriteGateNet(exp._exp_run(exp.ExpSpec("GJ", 11)), p)
    with torch.no_grad():
        model.gate_input.weight.fill_(0.1)
        model.gate_history.weight.fill_(0.05)
    generator = torch.Generator().manual_seed(9)
    x = torch.rand(3, p.steps, p.input_channels, generator=generator)
    lengths = torch.tensor([16, 13, 10], dtype=torch.long)
    ids = np.asarray(["a", "b", "c"])

    with torch.no_grad():
        learned = model(x, lengths)
        mean_gate = exp._intervention_forward(model, x, lengths, "A2", ids)

    for i, length in enumerate(lengths.tolist()):
        expected = learned["gate_m"][i, :length].mean()
        actual = mean_gate["gate_m"][i, :length]
        assert torch.allclose(actual, torch.full_like(actual, expected))
        assert torch.allclose(actual.mean(), expected)


def test_a3_preserves_gate_multiset_but_changes_alignment() -> None:
    p = _small_protocol()
    model = exp.ContextWriteGateNet(exp._exp_run(exp.ExpSpec("GJ", 11)), p)
    with torch.no_grad():
        model.gate_input.weight.fill_(0.2)
        model.gate_history.weight.fill_(0.1)
    generator = torch.Generator().manual_seed(19)
    x = torch.rand(2, p.steps, p.input_channels, generator=generator)
    lengths = torch.tensor([16, 12], dtype=torch.long)
    ids = np.asarray(["sample-a", "sample-b"])

    with torch.no_grad():
        learned = model(x, lengths)
        shuffled = exp._intervention_forward(
            model, x, lengths, "A3", ids, shuffle_seed=101
        )

    for i, length in enumerate(lengths.tolist()):
        before = torch.sort(learned["gate_m"][i, :length]).values
        after = torch.sort(shuffled["gate_m"][i, :length]).values
        assert torch.allclose(before, after)


def test_a4b_preserves_first_pass_mean_history_contribution() -> None:
    p = _small_protocol()
    model = exp.ContextWriteGateNet(exp._exp_run(exp.ExpSpec("GJ", 11)), p)
    with torch.no_grad():
        model.gate_history.weight.fill_(0.2)
    generator = torch.Generator().manual_seed(29)
    x = torch.rand(2, p.steps, p.input_channels, generator=generator)
    lengths = torch.tensor([16, 11], dtype=torch.long)
    ids = np.asarray(["sample-a", "sample-b"])

    with torch.no_grad():
        first = model(x, lengths)
        controlled = exp._intervention_forward(model, x, lengths, "A4b", ids)

    for i, length in enumerate(lengths.tolist()):
        expected = first["gate_history_term"][i, :length].mean()
        reconstructed = (
            torch.logit(controlled["gate_g"][i, :length].clamp(1e-6, 1 - 1e-6))
            - controlled["gate_input_term"][i, :length]
            - model.gate_bias
        )
        assert torch.allclose(
            reconstructed,
            torch.full_like(reconstructed, expected),
            atol=1e-5,
            rtol=0,
        )


def test_probe_gains_use_ordered_minus_matched_shuffled() -> None:
    import pandas as pd

    rows = []
    for aggregation, ba in (
        ("whole_count", 0.50),
        ("fixed250_ordered", 0.65),
        ("fixed250_shuffled", 0.57),
        ("relative10_ordered", 0.70),
        ("relative10_shuffled", 0.60),
    ):
        rows.append({
            "case": "GJ",
            "seed": 11,
            "layer": "L2",
            "state": "spike",
            "aggregation": aggregation,
            "decoder": "no_bias",
            "train_ba": ba,
            "val_ba": ba,
            "test_ba": ba,
        })
    gains = exp._probe_gains(pd.DataFrame(rows)).set_index("gain")
    assert gains.loc["G_order", "test_delta"] == pytest.approx(0.08)
    assert gains.loc["G_relative_order", "test_delta"] == pytest.approx(0.10)
    assert gains.loc["G_resolved", "test_delta"] == pytest.approx(0.15)
