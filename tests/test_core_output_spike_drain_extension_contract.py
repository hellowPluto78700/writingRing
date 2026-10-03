from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from core_benchmark_v1.model import BenchmarkNet, spike_readout
from core_benchmark_v1.protocol import Protocol, Run
from core_benchmark_v1.extensions.output_spike_drain.model import (
    drained_spike_output,
    drain_output_state,
    valid_output,
)
from core_benchmark_v1.extensions.output_spike_drain.protocol import (
    BETAS,
    DRAIN_BETA,
    MAX_DRAIN_STEPS,
    runs,
)


def test_spike_drain_manifest_contract() -> None:
    specs = runs()
    assert len(specs) == 33
    assert len({r.key for r in specs}) == 33
    assert BETAS == tuple(round(i / 10, 1) for i in range(11))
    assert DRAIN_BETA == 1.0
    assert MAX_DRAIN_STEPS == 1024


def test_valid_window_matches_core_spike_readout_exactly() -> None:
    torch.manual_seed(7)
    evidence = torch.randn(3, 9, 4)
    lengths = torch.tensor([9, 6, 3])
    beta, threshold, slope = 0.5, 0.5, 10.0
    old = spike_readout(evidence, lengths, beta, threshold, slope)
    new = valid_output(
        evidence, lengths, beta=beta, threshold=threshold, slope=slope
    )["spike"]
    assert torch.equal(old, new)


def test_padding_after_valid_length_does_not_change_endpoint_state() -> None:
    torch.manual_seed(9)
    prefix = torch.randn(2, 5, 3)
    tail = torch.randn(2, 4, 3)
    lengths = torch.tensor([5, 3])
    short = valid_output(
        prefix, lengths, beta=0.9, threshold=0.5, slope=10.0
    )
    long = valid_output(
        torch.cat([prefix, tail], dim=1),
        lengths,
        beta=0.9,
        threshold=0.5,
        slope=10.0,
    )
    assert torch.equal(short["valid_count"], long["valid_count"])
    assert torch.equal(short["final_membrane"], long["final_membrane"])


def test_known_positive_only_full_drain() -> None:
    theta = 0.5
    final = torch.tensor([[2.4 * theta, 0.4 * theta, -3.0 * theta]])
    out = drain_output_state(
        final, threshold=theta, slope=10.0, max_steps=32
    )
    torch.testing.assert_close(out["drain_count"], torch.tensor([[2.0, 0.0, 0.0]]))
    torch.testing.assert_close(
        out["residual_after_drain"],
        torch.tensor([[0.4 * theta, 0.4 * theta, -3.0 * theta]]),
    )
    assert out["k_required"].item() == 2
    assert out["cap_hit"].item() == 0


def test_drain_has_no_additional_leakage() -> None:
    theta = 0.5
    final = torch.tensor([[3.2 * theta]])
    drained = drain_output_state(
        final, threshold=theta, slope=10.0, max_steps=32
    )
    assert drained["drain_count"].item() == 3
    torch.testing.assert_close(
        drained["residual_after_drain"], torch.tensor([[0.2 * theta]])
    )


def test_drain_training_gradient_reaches_head_and_backbone() -> None:
    p = replace(
        Protocol(),
        profile="smoke",
        width=6,
        input_channels=3,
        total_channels=9,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=2,
    )
    run = Run("TEST_DRAIN", 11, "08_output_spike_drain", objective="wcce")
    model = BenchmarkNet(run, p)
    x = torch.randn(2, 16, 3)
    lengths = torch.tensor([16, 12])
    y = torch.tensor([0, 1])
    evidence = model(x, lengths)["evidence"]
    out = drained_spike_output(
        evidence,
        lengths,
        beta=0.9,
        threshold=p.threshold,
        slope=p.surrogate_slope,
        max_drain_steps=64,
    )
    torch.nn.functional.cross_entropy(out["drained_mean_logits"], y).backward()
    for parameter in (
        model.head.weight,
        model.layers[0].weight,
        model.layers[1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_spike_drain_slurm_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    slurm = (
        root
        / "core_benchmark_v1"
        / "extensions"
        / "output_spike_drain"
        / "slurm"
    )
    common = (slurm / "common.bash").read_text()
    assert "source /etc/profile" in common
    assert common.index("source /etc/profile") < common.index("module load conda/latest")
    for name in (
        "prepare.bash",
        "run_array.bash",
        "postprocess_array.bash",
        "finalize.bash",
    ):
        source = (slurm / name).read_text()
        assert "#SBATCH --time=01:00:00" in source
        assert "common.bash" in source
    run_array = (slurm / "run_array.bash").read_text()
    assert "#SBATCH --array=0-32%33" in run_array
    submit = (slurm / "submit.bash").read_text()
    assert '--dependency="afterok:$prepare"' in submit
    assert '--dependency="afterok:$train"' in submit
    assert '--dependency="afterok:$postprocess"' in submit
