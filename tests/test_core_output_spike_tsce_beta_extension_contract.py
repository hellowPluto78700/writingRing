from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from core_benchmark_v1.model import BenchmarkNet, valid_mask
from core_benchmark_v1.protocol import Protocol
from core_benchmark_v1.extensions.output_spike_drain.model import valid_output
from core_benchmark_v1.extensions.output_spike_tsce_beta.model import (
    native_count,
    spike_tsce_loss,
    spike_tsce_output,
)
from core_benchmark_v1.extensions.output_spike_tsce_beta.protocol import (
    BETAS,
    DRAIN_BETA,
    MAX_DRAIN_STEPS,
    MODES,
    ExtensionRun,
    runs,
)
from core_benchmark_v1.extensions.output_spike_tsce_beta.runner import core_run


def smoke_protocol() -> Protocol:
    return replace(
        Protocol(),
        profile="smoke",
        width=6,
        input_channels=3,
        total_channels=9,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=2,
    )


def test_manifest_has_paired_66_run_matrix() -> None:
    specs = runs()
    assert len(specs) == 66
    assert len({spec.key for spec in specs}) == 66
    assert BETAS == tuple(round(i / 10, 1) for i in range(11))
    assert MODES == ("valid", "drain")
    assert DRAIN_BETA == 1.0
    assert MAX_DRAIN_STEPS == 1024
    assert ExtensionRun("valid", 0.5, 11).key == "TSCE_VALID_B050__seed11"
    assert ExtensionRun("drain", 0.5, 11).key == "TSCE_DRAIN_B050__seed11"


def test_mode_and_beta_do_not_change_paired_parameter_initialization() -> None:
    p = smoke_protocol()
    reference = BenchmarkNet(core_run(ExtensionRun("valid", 0.0, 11)), p)
    for spec in (
        ExtensionRun("valid", 1.0, 11),
        ExtensionRun("drain", 0.0, 11),
        ExtensionRun("drain", 1.0, 11),
    ):
        candidate = BenchmarkNet(core_run(spec), p)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)


def test_valid_output_is_exact_established_output_dynamics() -> None:
    torch.manual_seed(3)
    evidence = torch.randn(3, 9, 4)
    lengths = torch.tensor([9, 6, 3])
    expected = valid_output(evidence, lengths, beta=0.7, threshold=0.5, slope=10.0)
    actual = spike_tsce_output(
        evidence,
        lengths,
        beta=0.7,
        threshold=0.5,
        slope=10.0,
        max_drain_steps=64,
        include_drain=False,
    )
    assert torch.equal(expected["spike"], actual["valid_trajectory"])
    assert torch.equal(expected["valid_count"], actual["valid_count"])
    assert torch.equal(actual["valid_count"], native_count(actual, "valid"))


def test_valid_tsce_matches_core_sample_balanced_temporal_reduction() -> None:
    trajectory = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ]
    )
    lengths = torch.tensor([3, 2])
    y = torch.tensor([0, 1])
    output = {"valid_trajectory": trajectory, "drained_trajectory": trajectory}
    targets = y[:, None].expand(-1, trajectory.shape[1])
    losses = F.cross_entropy(trajectory.transpose(1, 2), targets, reduction="none")
    expected = ((losses * valid_mask(lengths, 3)).sum(1) / lengths).mean()
    actual = spike_tsce_loss(output, lengths, y, "valid")
    torch.testing.assert_close(actual, expected)


def test_drain_is_attached_only_to_final_valid_tsce_timestep() -> None:
    theta = 0.5
    evidence = torch.tensor([[[1.2], [0.0], [0.0]], [[0.6], [0.0], [99.0]]])
    lengths = torch.tensor([3, 2])
    output = spike_tsce_output(
        evidence,
        lengths,
        beta=1.0,
        threshold=theta,
        slope=10.0,
        max_drain_steps=32,
        include_drain=True,
    )
    delta = output["drained_trajectory"] - output["valid_trajectory"]
    assert torch.equal(delta[0, :2], torch.zeros_like(delta[0, :2]))
    assert torch.equal(delta[1, 0], torch.zeros_like(delta[1, 0]))
    assert torch.equal(delta[1, 2], torch.zeros_like(delta[1, 2]))
    torch.testing.assert_close(delta[0, 2], output["drain_count"][0])
    torch.testing.assert_close(delta[1, 1], output["drain_count"][1])
    torch.testing.assert_close(output["drained_trajectory"].sum(1), output["total_count"])
    assert torch.equal(output["total_count"], native_count(output, "drain"))


def test_drain_tsce_keeps_original_valid_length_denominator() -> None:
    valid = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    drained = valid.clone()
    drained[0, 2] += torch.tensor([3.0, 0.0])
    output = {"valid_trajectory": valid, "drained_trajectory": drained}
    lengths = torch.tensor([3])
    y = torch.tensor([0])
    per_step = F.cross_entropy(
        drained.transpose(1, 2), y[:, None].expand(-1, 3), reduction="none"
    )
    expected = per_step.sum() / 3
    actual = spike_tsce_loss(output, lengths, y, "drain")
    torch.testing.assert_close(actual, expected)


def test_spike_tsce_gradient_reaches_head_and_hidden_layers_in_both_modes() -> None:
    p = smoke_protocol()
    x = torch.randn(2, 16, 3)
    lengths = torch.tensor([16, 12])
    y = torch.tensor([0, 1])
    for mode in MODES:
        spec = ExtensionRun(mode, 0.9, 11)
        model = BenchmarkNet(core_run(spec), p)
        evidence = model(x, lengths)["evidence"]
        output = spike_tsce_output(
            evidence,
            lengths,
            beta=spec.beta,
            threshold=p.threshold,
            slope=p.surrogate_slope,
            max_drain_steps=64,
            include_drain=mode == "drain",
        )
        spike_tsce_loss(output, lengths, y, mode).backward()
        for parameter in (
            model.head.weight,
            model.layers[0].weight,
            model.layers[1].weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_slurm_contract_uses_preflight_and_66_task_paired_array() -> None:
    root = Path(__file__).resolve().parents[1]
    slurm = root / "core_benchmark_v1" / "extensions" / "output_spike_tsce_beta" / "slurm"
    common = (slurm / "common.bash").read_text()
    assert "slurm_cpu_env.bash" in common
    assert "source /etc/profile" not in common
    run_array = (slurm / "run_array.bash").read_text()
    assert "#SBATCH --array=0-65%33" in run_array
    assert "#SBATCH --cpus-per-task=1" in run_array
    smoke = (slurm / "smoke.bash").read_text()
    assert "output_spike_tsce_beta" in smoke
    assert "import sys, torch, numpy, pandas" in smoke
    submit = (slurm / "submit.bash").read_text()
    assert '--dependency="afterok:$prepare"' in submit
    assert '--dependency="afterok:$smoke"' in submit
    assert '--dependency="afterok:$train"' in submit
