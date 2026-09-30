from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.protocol import Protocol
from scripts import experiment_15_context_dependent_write_gate as exp15
from scripts import experiment_15_1_width_capacity_ablation as exp


def test_width_matrix_and_task_counts() -> None:
    assert exp.WIDTHS == (64, 96, 128)
    assert exp.TRAIN_WIDTHS == (64, 96)
    assert len(exp.baseline_specs()) == 6
    assert len(exp.phase1_specs()) == 18
    assert len(exp.phase1_5_specs()) == 12

    assert {spec.width for spec in exp.baseline_specs()} == {64, 96}
    assert {spec.case for spec in exp.baseline_specs()} == {"B0"}
    assert {spec.width for spec in exp.phase1_specs()} == {64, 96}
    assert {spec.case for spec in exp.phase1_specs()} == {"C0", "GF", "GJ"}
    assert {spec.width for spec in exp.phase1_5_specs()} == {64, 96}
    assert {spec.case for spec in exp.phase1_5_specs()} == {"GF", "GJ"}

    assert all(spec.width != 128 for spec in exp.baseline_specs())
    assert all(spec.width != 128 for spec in exp.phase1_specs())
    assert all(spec.width != 128 for spec in exp.phase1_5_specs())


@pytest.mark.parametrize("width", [64, 96, 128])
def test_width_protocol_changes_only_hidden_width(width: int) -> None:
    core = Protocol()
    protocol = exp.make_width_protocol(core, width)
    core_payload = asdict(core)
    width_payload = asdict(protocol)
    assert width_payload["width"] == width
    for key, value in core_payload.items():
        if key != "width":
            assert width_payload[key] == value
    protocol.validate()


def test_width_protocol_rejects_unplanned_width() -> None:
    with pytest.raises(ValueError):
        exp.make_width_protocol(Protocol(), 32)


def test_parameter_counts_match_architecture() -> None:
    rows = {row["width"]: row for row in exp.parameter_counts()}
    assert rows[64]["baseline_parameters"] == 6784
    assert rows[96]["baseline_parameters"] == 13248
    assert rows[128]["baseline_parameters"] == 21760
    assert rows[64]["gate_parameters"] == 129
    assert rows[96]["gate_parameters"] == 193
    assert rows[128]["gate_parameters"] == 257

    for width in exp.WIDTHS:
        protocol = exp.make_width_protocol(Protocol(), width)
        baseline = BenchmarkNet(
            exp._run(exp.WidthSpec(width, "B0", 11)),
            protocol,
        )
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
        assert baseline_count == rows[width]["baseline_parameters"]

        gated = exp15.ContextWriteGateNet(
            exp._run(exp.WidthSpec(width, "GJ", 11)),
            protocol,
        )
        gated_count = sum(parameter.numel() for parameter in gated.parameters())
        assert gated_count == rows[width]["gated_total_parameters"]


@pytest.mark.parametrize("width", [64, 96])
def test_gate_initialization_remains_function_preserving(width: int) -> None:
    protocol = exp.make_width_protocol(Protocol(), width)
    run = exp._run(exp.WidthSpec(width, "GJ", 11))
    baseline = BenchmarkNet(run, protocol)
    gated = exp15.ContextWriteGateNet(run, protocol)
    gated.load_baseline_state(baseline.state_dict())

    generator = torch.Generator().manual_seed(151)
    x = torch.rand(3, 12, protocol.input_channels, generator=generator)
    lengths = torch.tensor([12, 9, 7], dtype=torch.long)

    with torch.no_grad():
        baseline_out = baseline(x, lengths)
        gated_out = gated(x, lengths)

    valid = torch.arange(12)[None, :] < lengths[:, None]
    assert torch.allclose(
        gated_out["gate_m"][valid],
        torch.ones_like(gated_out["gate_m"][valid]),
        atol=1e-6,
        rtol=0,
    )
    assert torch.allclose(
        baseline_out["evidence"],
        gated_out["evidence"],
        atol=1e-6,
        rtol=0,
    )
    assert torch.equal(baseline_out["spike"][0], gated_out["spike"][0])
    assert torch.equal(baseline_out["spike"][1], gated_out["spike"][1])


def test_width_specific_run_keys_are_unique_within_training_phases() -> None:
    baseline_keys = [spec.key for spec in exp.baseline_specs()]
    phase1_keys = [spec.key for spec in exp.phase1_specs()]
    phase15_keys = [spec.key for spec in exp.phase1_5_specs()]

    assert len(baseline_keys) == len(set(baseline_keys))
    assert len(phase1_keys) == len(set(phase1_keys))
    assert len(phase15_keys) == len(set(phase15_keys))
    assert set(baseline_keys).isdisjoint(phase1_keys)
    assert set(phase15_keys).issubset(set(phase1_keys))

    assert exp.WidthSpec(64, "B0", 11).key == "H64__B0__seed11"
    assert exp.WidthSpec(96, "GJ", 37).key == "H96__GJ__seed37"
