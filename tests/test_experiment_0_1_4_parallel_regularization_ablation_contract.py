from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import experiment_0_1_4_parallel_regularization_ablation as exp014


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_0_1_4_parallel_regularization_ablation.ipynb"
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_0_1_4_parallel_regularization_ablation_cpu_array.bash"
FINALIZE_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_0_1_4_parallel_regularization_ablation_cpu.bash"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_0_1_4_parallel_regularization_ablation_cpu.bash"


def test_run_matrix_is_exactly_180_binary_30epoch_runs() -> None:
    specs = exp014.run_specs()
    assert len(specs) == 180
    assert len({spec.key for spec in specs}) == 180
    assert exp014.EPOCHS == 30
    assert exp014.SEEDS == (11, 23, 37, 53, 71)
    assert exp014.OBJECTIVES == ("whole_count_ce", "timestep_ce")
    assert exp014.VARIANT == "binary"
    assert exp014.HIDDEN_CAP == 1
    assert exp014.CONDITIONS == (
        "no_reg",
        "original_reg",
        "temporal_norm",
        "tau_gain_norm",
        "warmup",
        "all_fixes",
    )
    assert {spec.architecture for spec in specs} == set(exp014.DIRECT_ARCHITECTURES)


def test_conditions_are_strictly_paired_on_model_and_loader_seeds() -> None:
    reference = exp014.RunSpec("no_reg", "short_mid", "whole_count_ce", 23)
    for condition in exp014.CONDITIONS:
        spec = exp014.RunSpec(condition, "short_mid", "whole_count_ce", 23)
        assert exp014.paired_seed(spec, "model_init") == exp014.paired_seed(reference, "model_init")
        assert exp014.paired_seed(spec, "train_loader") == exp014.paired_seed(reference, "train_loader")


def _toy_trajectory() -> dict[str, object]:
    return {
        "hidden_spikes": (torch.zeros(1, 2, 1),),
        "hidden_pre_reset": (torch.ones(1, 2, 1),),
        "output_spikes": torch.zeros(1, 2, 2),
    }


def test_original_reg_matches_exp012_definition_for_binary_hidden_spikes() -> None:
    trajectory = {
        "hidden_spikes": (
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]]),
            torch.tensor([[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]]),
        ),
        "hidden_pre_reset": (
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [1.0, 2.0]]]),
            torch.tensor([[[2.0, 1.0], [4.0, 3.0]], [[1.0, 2.0], [2.0, 1.0]]]),
        ),
        "output_spikes": torch.zeros(2, 2, 2),
    }
    lengths = torch.tensor([2, 1])
    alphas = (torch.tensor([0.5, 0.75]), torch.tensor([0.5, 0.75]))
    p2, a1, _ = exp014.regularization_terms(
        trajectory, lengths, alphas, "original_reg", tau=0.5
    )
    expected_p2, expected_a1 = exp014.exp012.regularization_terms(
        trajectory, lengths, hidden_cap=1, tau=0.5
    )
    assert torch.allclose(p2, expected_p2, atol=1e-7)
    assert torch.allclose(a1, expected_a1, atol=1e-7)


def test_temporal_normalization_converts_ew_sum_to_ew_mean() -> None:
    trajectory = _toy_trajectory()
    lengths = torch.tensor([2])
    alphas = (torch.tensor([0.5]),)
    original_p2, _, _ = exp014.regularization_terms(
        trajectory, lengths, alphas, "original_reg", tau=0.5
    )
    normalized_p2, _, _ = exp014.regularization_terms(
        trajectory, lengths, alphas, "temporal_norm", tau=0.5
    )
    assert torch.allclose(original_p2, torch.tensor(2.25), atol=1e-7)
    assert torch.allclose(normalized_p2, torch.tensor(1.0), atol=1e-7)


def test_tau_gain_normalization_only_rescales_p2_penalty_path() -> None:
    trajectory = _toy_trajectory()
    lengths = torch.tensor([2])
    alphas = (torch.tensor([0.75]),)
    tau_gain_p2, a1, _ = exp014.regularization_terms(
        trajectory, lengths, alphas, "tau_gain_norm", tau=0.0
    )
    all_fixes_p2, all_fixes_a1, _ = exp014.regularization_terms(
        trajectory, lengths, alphas, "all_fixes", tau=0.5
    )
    assert torch.allclose(tau_gain_p2, torch.tensor(0.0625), atol=1e-7)
    assert torch.allclose(all_fixes_p2, torch.tensor(0.0625), atol=1e-7)
    assert torch.allclose(a1, torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(all_fixes_a1, torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(trajectory["hidden_pre_reset"][0], torch.ones(1, 2, 1))


def test_warmup_and_no_reg_lambda_scales_are_exact() -> None:
    assert exp014._lambda_scale("no_reg", 1) == 0.0
    assert exp014._lambda_scale("original_reg", 1) == 1.0
    assert exp014._lambda_scale("temporal_norm", 1) == 1.0
    assert exp014._lambda_scale("tau_gain_norm", 1) == 1.0
    assert exp014._lambda_scale("warmup", 1) == 0.1
    assert exp014._lambda_scale("warmup", 5) == 0.5
    assert exp014._lambda_scale("warmup", 10) == 1.0
    assert exp014._lambda_scale("warmup", 30) == 1.0
    assert exp014._lambda_scale("all_fixes", 5) == 0.5


def test_all_fixes_manifest_combines_only_the_intended_three_repairs() -> None:
    manifest = exp014._condition_manifest("all_fixes")
    assert manifest["potential_source"] == "hidden_pre_reset_membrane"
    assert manifest["potential_layers"] == "all hidden spiking layers only"
    assert manifest["activity_layers"] == "all hidden spiking layers only"
    assert manifest["output_layer_regularized"] is False
    assert manifest["temporal_normalization"] is True
    assert manifest["tau_gain_normalization"] is True
    assert manifest["warmup"] is True
    assert manifest["warmup_epochs"] == 10
    assert manifest["lambda_p2_max"] == 0.01
    assert manifest["lambda_a1_max"] == 0.1


def test_finalizer_is_aggregation_only_and_uses_equal_budget_no_reg_pairs() -> None:
    source = inspect.getsource(exp014.finalize_experiment)
    for forbidden in ("train_one(", "run_one(", "optimizer", ".backward("):
        assert forbidden not in source
    assert "paired_condition_effects.csv" in source
    assert "history_long.csv" in source
    assert "gradient_diagnostics_summary.csv" in source
    assert '("no_reg", spec.architecture, spec.objective, spec.seed)' in source


def test_slurm_matches_repo_multi_cpu_contract_and_does_not_use_outputs_dir() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    finalize_text = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-179%50" in array_text
    assert "#SBATCH --cpus-per-task=1" in array_text
    assert "#SBATCH --cpus-per-task=1" in finalize_text
    assert "--output=exp0_1_4_regab_%A_%a.out" in array_text
    assert "--error=exp0_1_4_regab_%A_%a.err" in array_text
    for text in (array_text, finalize_text):
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        assert 'REPO_ROOT="${REPO_ROOT:-$PWD}"' in text
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert f"export {name}=1" in text
        assert "outputs/" not in text
        assert "mkdir -p outputs" not in text
    assert "run-one" in array_text
    assert "finalize" in finalize_text
    assert "--export=ALL" in submit_text
    assert 'afterok:${ARRAY_JOB}' in submit_text


def test_notebook_is_analysis_only_and_reads_finalized_artifacts() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "summary.csv",
        "history_long.csv",
        "paired_condition_effects.csv",
        "paired_condition_summary.csv",
        "gradient_diagnostics_summary.csv",
        "frozen_exp01_summary.csv",
        "delta_test_ba_vs_no_reg",
        "first_batch_reg_to_task_grad_ratio",
    ):
        assert token in joined
    for forbidden in (
        "optimizer.step(",
        ".backward()",
        "subprocess",
        "sbatch",
        "run-one",
        "train_one",
    ):
        assert forbidden not in joined
