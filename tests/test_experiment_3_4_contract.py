from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import experiment_3_4_causal_fixed250_objectives as exp34


def test_exp34_protocol_and_run_matrix() -> None:
    assert exp34.EXPERIMENT_ID == "experiment_3_4_causal_fixed250_objectives"
    assert exp34.PROTOCOL_VERSION == "causal_prefix_v1"
    assert exp34.OBJECTIVES == (
        "timestep_ce",
        "whole_fixed250_ce",
        "causal_fixed250_prefix_ce",
    )
    assert exp34.SEEDS == (11, 23, 101)
    assert exp34.EXPECTED_RUNS == 9
    assert len(exp34.run_specs()) == 9
    assert len({spec.key for spec in exp34.run_specs()}) == 9
    assert exp34.L1_WIDTH == 128
    assert exp34.L2_WIDTH == 128
    assert exp34.L1_SHIFTS == (2, 3, 4)
    assert exp34.L2_SHIFTS == (2, 3, 4)
    assert exp34.PREFIX_WEIGHT == 0.5
    assert exp34.FINAL_WEIGHT == 0.5


def test_exp34_partial_final_bin_is_retained() -> None:
    lengths = torch.tensor([1, 16, 17, 32, 33, 105, 256], dtype=torch.long)
    counts = exp34.valid_bin_counts(lengths, bin_steps=16, n_bins=16)
    assert counts.tolist() == [1, 1, 2, 2, 3, 7, 16]


def test_exp34_whole_and_causal_fixed_heads_are_parameter_matched() -> None:
    whole = exp34.build_initialized_model("whole_fixed250_ce", 12, 256, 64.0, 16, 11)
    causal = exp34.build_initialized_model("causal_fixed250_prefix_ce", 12, 256, 64.0, 16, 11)
    assert whole.head.weight.shape == causal.head.weight.shape == (12, 2048)
    assert whole.head.bias.shape == causal.head.bias.shape == (12,)
    assert whole.parameter_counts() == causal.parameter_counts()
    assert torch.equal(whole.head.weight, causal.head.weight)
    assert torch.equal(whole.head.bias, causal.head.bias)
    assert torch.equal(whole.f1.weight, causal.f1.weight)
    assert torch.equal(whole.f2.weight, causal.f2.weight)


def test_exp34_backbone_init_is_paired_across_all_objectives() -> None:
    models = [
        exp34.build_initialized_model(objective, 12, 256, 64.0, 16, 23)
        for objective in exp34.OBJECTIVES
    ]
    for model in models[1:]:
        assert torch.equal(models[0].f1.weight, model.f1.weight)
        assert torch.equal(models[0].f2.weight, model.f2.weight)


def test_exp34_last_prefix_equals_whole_linear_on_full_counts() -> None:
    whole = exp34.build_initialized_model("whole_fixed250_ce", 12, 256, 64.0, 16, 101)
    counts = torch.randn(4, 16, 128)
    prefix = whole.fixed_prefix_logits_from_counts(counts)
    direct = whole.head(counts.flatten(start_dim=1))
    assert torch.allclose(prefix[:, -1], direct, atol=1e-6, rtol=1e-6)


def test_exp34_prefix_logits_do_not_depend_on_future_bins() -> None:
    model = exp34.build_initialized_model("causal_fixed250_prefix_ce", 12, 256, 64.0, 16, 11)
    counts = torch.randn(3, 16, 128)
    changed = counts.clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 100.0
    before = model.fixed_prefix_logits_from_counts(counts)
    after = model.fixed_prefix_logits_from_counts(changed)
    assert torch.equal(before[:, :6], after[:, :6])


def test_exp34_causal_loss_uses_partial_final_bin_as_final_only() -> None:
    model = exp34.build_initialized_model("causal_fixed250_prefix_ce", 2, 48, 64.0, 16, 11)
    spikes = torch.zeros(1, 48, 128)
    spikes[:, :33, 0] = 1.0
    lengths = torch.tensor([33], dtype=torch.long)
    y = torch.tensor([1], dtype=torch.long)
    counts = model.fixed_counts(spikes, lengths)
    assert counts[0, 0, 0].item() == 16.0
    assert counts[0, 1, 0].item() == 16.0
    assert counts[0, 2, 0].item() == 1.0
    assert counts[0, 3 - 1, 0].item() == 1.0
    loss, final_logits, parts = model.loss_and_final_logits(spikes, lengths, y)
    prefix = model.fixed_prefix_logits_from_counts(counts)
    assert torch.allclose(final_logits, prefix[:, 2])
    assert torch.isfinite(loss)
    assert parts["prefix_loss"] == parts["prefix_loss"]
    assert parts["final_loss"] == parts["final_loss"]


def test_exp34_slurm_array_is_one_run_per_cpu() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (repo_root / "scripts/bash_script/SNN_Bash/run_exp_3_4_cpu_array.bash").read_text()
    finalizer = (repo_root / "scripts/bash_script/SNN_Bash/finalize_exp_3_4_cpu.bash").read_text()
    submitter = (repo_root / "scripts/bash_script/SNN_Bash/submit_exp_3_4_pipeline.bash").read_text()
    assert "#SBATCH --array=0-8%9" in runner
    assert "#SBATCH --cpus-per-task=1" in runner
    for variable in (
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
    ):
        assert variable in runner
    assert 'run-one --array-task-id "$TASK_ID" --device cpu' in runner
    assert "experiment_3_4_causal_fixed250_objectives finalize" in finalizer
    assert "--dependency=afterok:${EXP34_ARRAY}" in submitter


def test_exp34_notebook_declares_hypothesis_validation_and_conclusion() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "notebooks/experiment_3_4_causal_fixed250_objectives.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "Hypothesis" in markdown
    assert "Validation standard" in markdown
    assert "Conclusion" in markdown
    assert "partial final bin" in markdown
    assert "Relative10" in markdown
