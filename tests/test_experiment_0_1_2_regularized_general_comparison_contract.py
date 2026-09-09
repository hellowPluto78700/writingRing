from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import experiment_0_1_2_regularized_general_comparison as exp012


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_0_1_2_regularized_general_comparison.ipynb"
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_0_1_2_regularized_general_comparison_cpu_array.bash"
FINALIZE_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_0_1_2_regularized_general_comparison_cpu.bash"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_0_1_2_regularized_general_comparison_cpu.bash"


def test_run_matrix_is_exactly_60_new_regularized_direct_runs() -> None:
    specs = exp012.run_specs()
    assert len(specs) == 60
    assert len({spec.key for spec in specs}) == 60
    assert exp012.SEEDS == (11, 23, 37, 53, 71)
    assert exp012.OBJECTIVES == ("whole_count_ce", "timestep_ce")
    assert exp012.VARIANTS == (("binary", 1), ("multi_h", 31))
    assert {spec.architecture for spec in specs} == set(exp012.DIRECT_ARCHITECTURES)
    assert {spec.objective for spec in specs} == set(exp012.OBJECTIVES)
    assert {spec.variant for spec in specs} == {"binary", "multi_h"}
    assert all("reg_on" in spec.key for spec in specs)


def test_regularization_contract_is_hidden_only_and_fixed() -> None:
    manifest = exp012._regularization_manifest()
    assert exp012.REG_TAU == 0.99
    assert exp012.REG_LAMBDA_P2 == 0.01
    assert exp012.REG_LAMBDA_A1 == 0.1
    assert manifest["potential_term"] == "P2"
    assert manifest["potential_source"] == "pre_reset_membrane"
    assert manifest["output_layer_regularized"] is False
    assert "hidden" in str(manifest["potential_layers"])
    assert "hidden" in str(manifest["activity_layers"])
    assert "hidden_cap" in str(manifest["activity_normalization"])


def test_model_seed_and_state_dict_match_frozen_exp01_pair() -> None:
    spec = exp012.RunSpec("short_mid", "whole_count_ce", "multi_h", 31, 11)
    base = exp012.base_spec(spec)

    exp012.exp01.exp3.seed_all(exp012.paired_seed(spec, "model_init"))
    regularized = exp012.RegularizedMultiTauHierarchySNN(
        exp012.DIRECT_ARCHITECTURES[spec.architecture], 12, 64.0, spec.hidden_cap
    )
    exp012.exp01.exp3.seed_all(exp012.exp01.paired_seed(base, "model_init"))
    frozen = exp012.exp01.MultiTauHierarchySNN(
        exp012.exp01.hidden_shifts(base), 12, 64.0, spec.hidden_cap, True
    )

    assert regularized.state_dict().keys() == frozen.state_dict().keys()
    for name, value in regularized.state_dict().items():
        assert torch.allclose(value, frozen.state_dict()[name])


def test_p2_freezes_after_valid_endpoint() -> None:
    pre = torch.ones(2, 4, 1)
    spikes = torch.ones(2, 4, 1)
    trajectory: dict[str, object] = {
        "hidden_spikes": (spikes,),
        "hidden_pre_reset": (pre,),
    }
    lengths = torch.tensor([2, 4])
    p2, a1 = exp012.regularization_terms(trajectory, lengths, hidden_cap=1, tau=0.5)

    expected_u0 = 1.5
    expected_u1 = 1.875
    expected_p2 = (expected_u0**2 + expected_u1**2) / 2.0
    assert torch.allclose(p2, torch.tensor(expected_p2), atol=1e-6)
    assert torch.allclose(a1, torch.tensor(1.0), atol=1e-6)


def test_a1_is_cap_normalized_between_binary_and_multih() -> None:
    lengths = torch.tensor([3])
    pre = torch.zeros(1, 3, 2)
    binary: dict[str, object] = {
        "hidden_spikes": (torch.ones(1, 3, 2),),
        "hidden_pre_reset": (pre,),
    }
    multih: dict[str, object] = {
        "hidden_spikes": (torch.full((1, 3, 2), 31.0),),
        "hidden_pre_reset": (pre,),
    }
    _, binary_a1 = exp012.regularization_terms(binary, lengths, hidden_cap=1)
    _, multih_a1 = exp012.regularization_terms(multih, lengths, hidden_cap=31)
    assert torch.allclose(binary_a1, multih_a1, atol=1e-7)
    assert torch.allclose(binary_a1, torch.tensor(1.0), atol=1e-7)


def test_finalizer_is_aggregation_only() -> None:
    source = inspect.getsource(exp012.finalize_experiment)
    for forbidden in ("train_one(", "run_one(", "optimizer", ".backward("):
        assert forbidden not in source
    assert "_load_frozen_base_runs" in source
    assert "baseline_results.csv" in source


def test_slurm_and_notebook_follow_multi_cpu_analysis_only_contract() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    finalize_text = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-59%50" in array_text
    assert "#SBATCH --cpus-per-task=1" in array_text
    assert "#SBATCH --cpus-per-task=1" in finalize_text
    for text in (array_text, finalize_text):
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert f"export {name}=1" in text
    assert "run-one" in array_text
    assert "finalize" in finalize_text
    assert 'afterok:${ARRAY_JOB}' in submit_text
    assert "BASELINE_JOB" not in submit_text

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
        "comparison_summary.csv",
        "paired_regularization_effects.csv",
        "regularization_objective_interactions.csv",
        "regularization_capacity_interactions.csv",
        "regularization_architecture_interactions.csv",
        "delta_test_ba_reg_on_minus_off",
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
