from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import experiment_0_1_3_doc_faithful_regularization as exp013


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_0_1_3_doc_faithful_regularization.ipynb"
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_0_1_3_doc_faithful_regularization_cpu_array.bash"
FINALIZE_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_0_1_3_doc_faithful_regularization_cpu.bash"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_0_1_3_doc_faithful_regularization_cpu.bash"


def test_run_matrix_is_exactly_30_binary_direct_runs() -> None:
    specs = exp013.run_specs()
    assert len(specs) == 30
    assert len({spec.key for spec in specs}) == 30
    assert exp013.SEEDS == (11, 23, 37, 53, 71)
    assert exp013.OBJECTIVES == ("whole_count_ce", "timestep_ce")
    assert exp013.VARIANT == "binary"
    assert exp013.HIDDEN_CAP == 1
    assert {spec.architecture for spec in specs} == set(exp013.DIRECT_ARCHITECTURES)
    assert {spec.objective for spec in specs} == set(exp013.OBJECTIVES)
    assert all("doc_reg_on" in spec.key for spec in specs)


def test_exp01_training_and_architecture_parameters_are_preserved() -> None:
    assert exp013.HIDDEN_WIDTH == exp013.exp01.HIDDEN_WIDTH
    assert exp013.OUTPUT_CAP == exp013.exp01.OUTPUT_CAP
    assert exp013.BATCH_SIZE == exp013.exp01.BATCH_SIZE
    assert exp013.EPOCHS == exp013.exp01.EPOCHS
    assert exp013.LR == exp013.exp01.LR
    assert exp013.WEIGHT_DECAY == exp013.exp01.WEIGHT_DECAY
    assert exp013.DIRECT_ARCHITECTURES == exp013.exp01.DIRECT_ARCHITECTURES
    assert exp013.OBJECTIVES == exp013.exp01.OBJECTIVES
    assert exp013.SEEDS == exp013.exp01.SEEDS


def test_doc_faithful_regularization_manifest() -> None:
    manifest = exp013._regularization_manifest()
    assert exp013.REG_TAU == 0.99
    assert exp013.REG_LAMBDA_P2 == 0.01
    assert exp013.REG_LAMBDA_A1 == 0.1
    assert manifest["configuration"] == "SAE-Dense"
    assert manifest["binary_only"] is True
    assert manifest["potential_source"] == "post_reset_membrane"
    assert manifest["potential_layers"] == "all_spiking_layers_including_output"
    assert manifest["activity_layers"] == "last_hidden_latent_layer_only"
    assert manifest["activity_signal"] == "binary_spike_output"
    assert manifest["lambda_p1"] == 0.0
    assert manifest["lambda_l2"] == 0.0
    assert "direct_sum" in str(manifest["potential_reduction"])


def test_model_seed_and_state_dict_match_frozen_exp01_binary_pair() -> None:
    spec = exp013.RunSpec("short_mid", "whole_count_ce", 11)
    base = exp013.base_spec(spec)

    exp013.exp01.exp3.seed_all(exp013.paired_seed(spec, "model_init"))
    regularized = exp013.DocFaithfulRegularizedSNN(
        exp013.DIRECT_ARCHITECTURES[spec.architecture], 12, 64.0
    )
    exp013.exp01.exp3.seed_all(exp013.exp01.paired_seed(base, "model_init"))
    frozen = exp013.exp01.MultiTauHierarchySNN(
        exp013.exp01.hidden_shifts(base), 12, 64.0, 1, True
    )

    assert regularized.state_dict().keys() == frozen.state_dict().keys()
    for name, value in regularized.state_dict().items():
        assert torch.allclose(value, frozen.state_dict()[name])


def test_p2_uses_post_reset_all_spiking_layers_and_freezes_endpoint() -> None:
    hidden_1 = torch.tensor([[[1.0], [1.0], [100.0]], [[1.0], [1.0], [1.0]]])
    hidden_2 = torch.tensor([[[2.0], [2.0], [100.0]], [[2.0], [2.0], [2.0]]])
    output = torch.tensor([[[3.0], [3.0], [100.0]], [[3.0], [3.0], [3.0]]])
    latent_spikes = torch.tensor([[[1.0], [0.0], [1.0]], [[1.0], [1.0], [0.0]]])
    ignored_first_layer_spikes = torch.full((2, 3, 1), 99.0)
    trajectory: dict[str, object] = {
        "hidden_spikes": (ignored_first_layer_spikes, latent_spikes),
        "hidden_post_reset": (hidden_1, hidden_2),
        "output_post_reset": output,
    }
    lengths = torch.tensor([2, 3])
    p2, a1, components = exp013.regularization_terms(
        trajectory, lengths, tau=0.5
    )

    expected_h1 = 1.5**2 + 1.75**2
    expected_h2 = 3.0**2 + 3.5**2
    expected_out = 4.5**2 + 5.25**2
    expected_p2 = expected_h1 + expected_h2 + expected_out
    expected_a1 = 1.0 / 2.0 + 2.0 / 3.0

    assert len(components) == 3
    assert torch.allclose(components[0], torch.tensor(expected_h1), atol=1e-6)
    assert torch.allclose(components[1], torch.tensor(expected_h2), atol=1e-6)
    assert torch.allclose(components[2], torch.tensor(expected_out), atol=1e-6)
    assert torch.allclose(p2, torch.tensor(expected_p2), atol=1e-6)
    assert torch.allclose(a1, torch.tensor(expected_a1), atol=1e-6)


def test_a1_has_no_cap_or_neuron_layer_normalization() -> None:
    latent_spikes = torch.ones(1, 2, 3)
    trajectory: dict[str, object] = {
        "hidden_spikes": (latent_spikes,),
        "hidden_post_reset": (torch.zeros(1, 2, 3),),
        "output_post_reset": torch.zeros(1, 2, 2),
    }
    p2, a1, _ = exp013.regularization_terms(
        trajectory, torch.tensor([2]), tau=0.99
    )
    assert torch.allclose(p2, torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(a1, torch.tensor(3.0), atol=1e-7)


def test_finalizer_is_aggregation_only_and_binary_paired() -> None:
    source = inspect.getsource(exp013.finalize_experiment)
    for forbidden in ("train_one(", "run_one(", "optimizer", ".backward("):
        assert forbidden not in source
    assert "_load_frozen_binary_direct_runs" in source
    assert "paired_regularization_effects.csv" in source
    assert "regularization_capacity_interactions" not in source


def test_slurm_and_notebook_follow_multi_cpu_analysis_only_contract() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    finalize_text = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-29%30" in array_text
    assert "#SBATCH --cpus-per-task=1" in array_text
    assert "#SBATCH --cpus-per-task=1" in finalize_text
    for text in (array_text, finalize_text):
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
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
        "regularization_architecture_interactions.csv",
        "delta_test_ba_doc_reg_minus_off",
        "train_reg_to_task_ratio",
        "train_p2_output",
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
