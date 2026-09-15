from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import experiment_7_2_6_1_sum_vs_mean_ce as exp7261
from scripts import experiment_7_2_6_output_readout_loss_shaping as exp726


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_narrow_task_only_three_seed_control() -> None:
    assert exp7261.ARCHITECTURE == "234x234"
    assert exp7261.REGULARIZATION == "task_only"
    assert exp7261.SEEDS == (11, 23, 37)
    assert len(exp7261.specs()) == 3
    assert exp7261.CONDITIONS == (exp7261.SUM_REUSE, exp7261.MEAN_NEW)


def test_mean_run_reuses_exact_exp726_c0_pairing() -> None:
    spec = exp7261.RunSpec(23)
    assert spec.c0_spec.regularization == "task_only"
    assert spec.c0_spec.seed == 23
    assert spec.c0_spec.condition == exp726.E2E_ANALOG
    assert exp726._reference_seed(spec.source, "model_init") == exp726._reference_seed(spec.c0_spec.source, "model_init")
    assert exp726._reference_seed(spec.source, "train_loader") == exp726._reference_seed(spec.c0_spec.source, "train_loader")


def test_fixed_model_sum_mean_argmax_equivalence() -> None:
    torch.manual_seed(7)
    evidence = torch.randn(5, 17, 12)
    lengths = torch.tensor([17, 16, 12, 9, 5])
    summed = exp7261._aggregate_evidence(evidence, lengths, "sum")
    meaned = exp7261._aggregate_evidence(evidence, lengths, "mean")
    assert torch.allclose(summed, meaned * lengths.to(evidence.dtype).unsqueeze(1))
    assert torch.equal(summed.argmax(1), meaned.argmax(1))


def test_output_projection_remains_bias_free() -> None:
    model = exp726.PairedShapingSNN(exp726.E2E_ANALOG, 12, 64.0)
    assert model.output_linear.bias is None


def test_notebook_is_summary_only() -> None:
    path = REPO_ROOT / "notebooks" / "experiment_7_2_6_1_sum_vs_mean_ce.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for required in (
        "native_performance_summary.csv",
        "l2_probe_summary.csv",
        "paired_delta_summary.csv",
        "readout_equivalence_summary.csv",
        "scaling_diagnostics_summary.csv",
        "training_history_summary.csv",
    ):
        assert required in text
    for forbidden in (
        "torch.load",
        "run_mean(",
        "run_probe(",
        "sbatch",
        "subprocess",
        "_runs.csv",
    ):
        assert forbidden not in text


def test_slurm_pipeline_is_three_way_cpu_parallel() -> None:
    base = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    for filename in (
        "run_exp_7_2_6_1_mean_ce_cpu_array.bash",
        "run_exp_7_2_6_1_probe_cpu_array.bash",
    ):
        text = (base / filename).read_text(encoding="utf-8")
        assert "#SBATCH --array=0-2%3" in text
        assert "#SBATCH --cpus-per-task=1" in text
        assert "--device cpu --threads 1" in text
    submit = (base / "submit_exp_7_2_6_1_cpu.bash").read_text(encoding="utf-8")
    assert submit.count("afterok:") == 2
    assert "finalize_exp_7_2_6_1_cpu.bash" in submit
