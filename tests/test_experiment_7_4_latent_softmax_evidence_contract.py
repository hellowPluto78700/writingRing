from pathlib import Path

import torch

from scripts import experiment_7_4_latent_softmax_evidence as exp74


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_task_counts() -> None:
    assert exp74.ARCHITECTURE == "234x234"
    assert exp74.SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp74.SEEDS == (11, 23, 37)
    assert exp74.TEMPERATURE == 1.0
    assert len(exp74.HEAD_CASES) == 4
    assert len(exp74.run_specs()) == 12
    assert len({spec.method for spec in exp74.run_specs()}) == 4


def test_head_parameterization_and_bias_contract() -> None:
    direct = exp74.LatentEvidenceHead("direct", 12)
    two_linear = exp74.LatentEvidenceHead("two_linear", 12)
    softmax = exp74.LatentEvidenceHead("softmax", 12)
    softmax_rms = exp74.LatentEvidenceHead("softmax_rms", 12)

    assert direct.direct is not None
    assert direct.direct.bias is None
    assert sum(p.numel() for p in direct.parameters()) == 128 * 12

    for head in (two_linear, softmax, softmax_rms):
        assert head.to_latent is not None
        assert head.to_class is not None
        assert head.to_latent.bias is None
        assert head.to_class.bias is None
        assert sum(p.numel() for p in head.parameters()) == 128 * 128 + 128 * 12


def test_softmax_and_rms_head_semantics() -> None:
    torch.manual_seed(7)
    z = torch.randn(3, 128)

    softmax = exp74.LatentEvidenceHead("softmax", 12)
    softmax_rms = exp74.LatentEvidenceHead("softmax_rms", 12)
    softmax_rms.load_state_dict(softmax.state_dict(), strict=True)

    _, aux_softmax = softmax(z, collect_diagnostics=True)
    _, aux_rms = softmax_rms(z, collect_diagnostics=True)

    assert torch.all(aux_softmax["softmax_max_prob"] > 0)
    assert torch.all(aux_softmax["softmax_max_prob"] <= 1)
    assert torch.all(aux_softmax["softmax_entropy"] >= 0)
    assert torch.all(aux_rms["magnitude_rms"] > 0)
    assert torch.allclose(aux_rms["magnitude_rms"], aux_rms["latent_rms"])


def test_two_linear_is_linear_when_weights_are_fixed() -> None:
    torch.manual_seed(9)
    head = exp74.LatentEvidenceHead("two_linear", 12)
    z1 = torch.randn(2, 128)
    z2 = torch.randn(2, 128)
    a = 1.7
    b = -0.4

    lhs, _ = head(a * z1 + b * z2)
    y1, _ = head(z1)
    y2, _ = head(z2)
    rhs = a * y1 + b * y2
    assert torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-5)


def test_slurm_array_and_dependency_chain() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    run = (root / "run_exp_7_4_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-11%12" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "MKL_NUM_THREADS=1" in run
    assert "--array-task-id" in run

    submit = (root / "submit_exp_7_4_cpu.bash").read_text()
    assert 'afterok:${array_job}' in submit
    assert "run_exp_7_4_cpu_array.bash" in submit
    assert "finalize_exp_7_4_cpu.bash" in submit


def test_plan_documents_primary_contrasts() -> None:
    text = (REPO_ROOT / "docs" / "plans" / "EXP7_4_LATENT_SOFTMAX_EVIDENCE.md").read_text()
    assert "B - A" in text
    assert "C - B" in text
    assert "D - C" in text
    assert "4 methods x 3 seeds = 12 jobs" in text
