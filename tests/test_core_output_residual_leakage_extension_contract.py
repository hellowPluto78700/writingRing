import torch
import torch.nn.functional as F

from core_benchmark_v1.model import BenchmarkNet, mean_logits
from core_benchmark_v1.protocol import Protocol, Run, runs as core_runs
from core_benchmark_v1.extensions.output_residual_leakage.model import (
    extension_loss,
    residual_leakage_output,
)
from core_benchmark_v1.extensions.output_residual_leakage.protocol import (
    BETAS,
    OBJECTIVES,
    SEEDS,
    ExtensionRun,
    runs,
)
from core_benchmark_v1.extensions.output_residual_leakage.runner import core_run


def test_extension_matrix_is_66_and_does_not_modify_core_matrix() -> None:
    extension = runs()
    assert len(extension) == 66
    assert {r.beta for r in extension} == set(BETAS)
    assert {r.objective for r in extension} == set(OBJECTIVES)
    assert {r.seed for r in extension} == set(SEEDS)
    assert len({r.key for r in extension}) == 66
    assert len(core_runs(Protocol())) == 36


def test_beta1_spike_plus_endpoint_residual_exactly_recovers_accumulator() -> None:
    evidence = torch.tensor(
        [
            [[1.4, -0.8], [0.1, 0.9], [1.2, -0.4], [0.7, 0.3]],
            [[-0.2, 1.8], [2.1, -1.2], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([4, 2])
    out = residual_leakage_output(evidence, lengths, beta=1.0, threshold=0.5, slope=25.0)
    expected = torch.stack([evidence[0, :4].sum(0), evidence[1, :2].sum(0)])
    torch.testing.assert_close(out["total_score"], expected, rtol=0, atol=2e-6)
    torch.testing.assert_close(out["analog_score"], expected, rtol=0, atol=0)


def test_endpoint_residual_is_added_only_at_each_samples_final_valid_index() -> None:
    evidence = torch.tensor(
        [
            [[0.2], [0.2], [0.2], [0.2]],
            [[0.1], [0.1], [0.0], [0.0]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([4, 2])
    out = residual_leakage_output(evidence, lengths, beta=0.7, threshold=0.5, slope=25.0)
    base = 0.5 * out["spike"]
    delta = out["trajectory"] - base
    assert torch.count_nonzero(delta[0, :3]) == 0
    assert torch.count_nonzero(delta[1, :1]) == 0
    assert torch.count_nonzero(delta[1, 2:]) == 0
    torch.testing.assert_close(delta[0, 3], out["final_residual"][0])
    torch.testing.assert_close(delta[1, 1], out["final_residual"][1])


def test_beta0_has_no_cross_timestep_residual_retention() -> None:
    evidence = torch.tensor([[[0.4], [0.0], [0.0]]], dtype=torch.float32)
    lengths = torch.tensor([3])
    out = residual_leakage_output(evidence, lengths, beta=0.0, threshold=0.5, slope=25.0)
    torch.testing.assert_close(out["residual"][0, 0, 0], torch.tensor(0.4), rtol=0, atol=1e-7)
    assert out["pre_reset"][0, 1, 0].item() == 0.0
    assert out["residual"][0, 1, 0].item() == 0.0


def test_beta1_wcce_loss_equals_core_o0_mean_logit_ce() -> None:
    torch.manual_seed(7)
    evidence = torch.randn(3, 6, 4)
    lengths = torch.tensor([6, 4, 5])
    y = torch.tensor([0, 2, 1])
    out = residual_leakage_output(evidence, lengths, beta=1.0, threshold=0.5, slope=25.0)
    actual = extension_loss(out, lengths, y, "wcce")
    expected = F.cross_entropy(mean_logits(evidence, lengths), y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)


def test_extension_uses_core_paired_initialization_streams() -> None:
    p = Protocol()
    for seed in SEEDS:
        ext = BenchmarkNet(core_run(ExtensionRun("wcce", 0.3, seed)), p)
        core = BenchmarkNet(Run("O0", seed, "01_objective", objective="wcce"), p)
        for name, value in ext.state_dict().items():
            torch.testing.assert_close(value, core.state_dict()[name], rtol=0, atol=0)


def test_beta1_wcce_evidence_gradient_equals_core_o0_gradient() -> None:
    torch.manual_seed(19)
    lengths = torch.tensor([7, 5, 6])
    y = torch.tensor([1, 0, 2])

    extension_evidence = torch.randn(3, 7, 4, requires_grad=True)
    core_evidence = extension_evidence.detach().clone().requires_grad_(True)

    out = residual_leakage_output(
        extension_evidence,
        lengths,
        beta=1.0,
        threshold=0.5,
        slope=25.0,
    )
    extension_value = extension_loss(out, lengths, y, "wcce")
    core_value = F.cross_entropy(mean_logits(core_evidence, lengths), y)

    extension_value.backward()
    core_value.backward()

    torch.testing.assert_close(extension_evidence.grad, core_evidence.grad, rtol=1e-5, atol=2e-6)

def test_slurm_workers_resolve_common_from_submit_root() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    slurm = root / "core_benchmark_v1" / "extensions" / "output_residual_leakage" / "slurm"
    workers = ("prepare.bash", "run_array.bash", "analyze_array.bash", "prune_array.bash", "finalize.bash")
    for name in workers:
        source = (slurm / name).read_text()
        assert '$(dirname "$0")' not in source
        assert "SLURM_SUBMIT_DIR" in source
        assert "common.bash" in source

    common = (slurm / "common.bash").read_text()
    assert 'REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"' in common

    submit = (slurm / "submit.bash").read_text()
    assert "export REPO_ROOT=" in submit
    assert "--export=ALL" in submit
