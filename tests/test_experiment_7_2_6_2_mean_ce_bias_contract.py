from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_6_output_readout_loss_shaping as exp726
from scripts import experiment_7_2_6_1_sum_vs_mean_ce as exp7261
from scripts import experiment_7_2_6_2_mean_ce_bias as exp7262


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_narrow_three_seed_bias_control() -> None:
    assert exp7262.ARCHITECTURE == "234x234"
    assert exp7262.REGULARIZATION == "task_only"
    assert exp7262.SEEDS == (11, 23, 37)
    assert len(exp7262.specs()) == 3
    assert exp7262.NATIVE_CONDITIONS == (
        exp7262.NO_BIAS_REUSE,
        exp7262.BIAS_TRAINED_WITH_BIAS,
        exp7262.BIAS_TRAINED_ZERO_BIAS,
    )


def test_bias_model_preserves_exact_hidden_and_w_initialization() -> None:
    spec = exp7262.RunSpec(23)
    seed = exp726._reference_seed(spec.source, "model_init")
    exp3.seed_all(seed)
    reference = exp726.PairedShapingSNN(exp726.E2E_ANALOG, 12, 64.0)
    biased = exp7262._build_bias_model(spec, 12, 64.0, torch.device("cpu"))
    for ref_layer, bias_layer in zip(reference.hidden_linears, biased.hidden_linears):
        assert torch.equal(ref_layer.weight, bias_layer.weight)
    assert torch.equal(reference.output_linear.weight, biased.output_linear.weight)
    assert reference.output_linear.bias is None
    assert biased.output_linear.bias is not None
    assert torch.count_nonzero(biased.output_linear.bias).item() == 0


def test_mean_of_per_timestep_biased_logits_is_w_mean_z_plus_b() -> None:
    torch.manual_seed(7)
    z = torch.randn(4, 11, 128)
    lengths = torch.tensor([11, 9, 6, 3])
    W = torch.randn(12, 128)
    b = torch.randn(12)
    evidence = torch.einsum("bth,kh->btk", z, W) + b
    mean_logits = exp7261._aggregate_evidence(evidence, lengths, "mean")
    sum_logits = exp7261._aggregate_evidence(evidence, lengths, "sum")
    mean_z = exp7261._aggregate_l2(z, lengths, "mean")
    expected = mean_z @ W.T + b
    assert torch.allclose(mean_logits, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sum_logits, mean_logits * lengths.to(z.dtype).unsqueeze(1), atol=1e-4, rtol=1e-5)
    assert torch.equal(sum_logits.argmax(1), mean_logits.argmax(1))


def test_pairing_reuses_exp7261_seed_stream() -> None:
    spec = exp7262.RunSpec(37)
    assert spec.baseline_spec.seed == 37
    assert exp726._reference_seed(spec.source, "model_init") == exp726._reference_seed(spec.c0_spec.source, "model_init")
    assert exp726._reference_seed(spec.source, "train_loader") == exp726._reference_seed(spec.c0_spec.source, "train_loader")


def test_notebook_is_summary_only() -> None:
    path = REPO_ROOT / "notebooks" / "experiment_7_2_6_2_mean_ce_bias.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for required in (
        "native_performance_summary.csv",
        "bias_effect_decomposition_summary.csv",
        "l2_probe_summary.csv",
        "l2_probe_delta_summary.csv",
        "bias_diagnostics_summary.csv",
        "bias_values.csv",
        "training_history_summary.csv",
    ):
        assert required in text
    for forbidden in (
        "torch.load",
        "run_bias(",
        "run_probe(",
        "sbatch",
        "subprocess",
        "_runs.csv",
    ):
        assert forbidden not in text


def test_slurm_pipeline_is_three_way_cpu_parallel() -> None:
    base = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    for filename in (
        "run_exp_7_2_6_2_bias_cpu_array.bash",
        "run_exp_7_2_6_2_probe_cpu_array.bash",
    ):
        text = (base / filename).read_text(encoding="utf-8")
        assert "#SBATCH --array=0-2%3" in text
        assert "#SBATCH --cpus-per-task=1" in text
        assert "--device cpu --threads 1" in text
    submit = (base / "submit_exp_7_2_6_2_cpu.bash").read_text(encoding="utf-8")
    assert submit.count("afterok:") == 2
    assert "finalize_exp_7_2_6_2_cpu.bash" in submit
