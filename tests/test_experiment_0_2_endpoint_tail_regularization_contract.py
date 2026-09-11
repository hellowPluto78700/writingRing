from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import experiment_0_2_endpoint_tail_regularization as exp02


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_0_2_endpoint_tail_regularization.ipynb"
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_0_2_endpoint_tail_cpu_array.bash"
FROZEN_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "eval_exp_0_2_frozen_exp01_cpu.bash"
FINALIZE_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_0_2_endpoint_tail_cpu.bash"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_0_2_endpoint_tail_cpu.bash"


def test_run_matrix_and_frozen_controls() -> None:
    specs = exp02.run_specs()
    frozen = exp02.frozen_specs()
    assert len(specs) == 72
    assert len({spec.key for spec in specs}) == 72
    assert len(frozen) == 18
    assert len({spec.key for spec in frozen}) == 18
    assert exp02.SEEDS == (11, 23, 37)
    assert exp02.OBJECTIVES == ("whole_count_ce", "timestep_ce")
    assert exp02.PROFILES == ("a1", "a1_p2", "tail", "a1_tail")
    assert {spec.architecture for spec in specs} == set(exp02.DIRECT_ARCHITECTURES)
    assert all(spec.profile == "none" for spec in frozen)
    assert exp02.EPOCHS == 50
    assert exp02.HIDDEN_CAP == exp02.OUTPUT_CAP == 1


def test_tail_rollout_is_endpoint_aligned_and_zero_after_each_endpoint() -> None:
    X = torch.arange(2 * 10 * 3, dtype=torch.float32).reshape(2, 10, 3)
    lengths = torch.tensor([4, 7])
    fs = 10.0
    rollout = exp02.endpoint_rollout_input(X, lengths, fs)
    assert rollout.shape[1] == 13
    assert torch.equal(rollout[0, :4], X[0, :4])
    assert torch.equal(rollout[1, :7], X[1, :7])
    assert torch.count_nonzero(rollout[0, 4:]) == 0
    assert torch.count_nonzero(rollout[1, 7:]) == 0


def test_tail_stage_masks_follow_each_samples_endpoint_not_padding_end() -> None:
    fs = 10.0
    lengths = torch.tensor([4, 7])
    masks = exp02.tail_stage_masks(lengths, n_steps=13, fs=fs)
    assert [int(mask[0].sum()) for mask in masks] == [2, 2, 2]
    assert [int(mask[1].sum()) for mask in masks] == [2, 2, 2]
    assert torch.nonzero(masks[0][0], as_tuple=False).flatten().tolist() == [4, 5]
    assert torch.nonzero(masks[0][1], as_tuple=False).flatten().tolist() == [7, 8]
    assert torch.nonzero(masks[2][0], as_tuple=False).flatten().tolist() == [8, 9]
    assert torch.nonzero(masks[2][1], as_tuple=False).flatten().tolist() == [11, 12]


def test_tail_activity_is_stage_weighted_and_layer_normalized() -> None:
    fs = 10.0
    lengths = torch.tensor([2])
    n_steps = 8
    layer1 = torch.zeros(1, n_steps, 2)
    layer2 = torch.zeros(1, n_steps, 2)
    # Stages are [2,4), [4,6), [6,8). Layer means become 0.5, 0.25, 0.5.
    layer1[:, 2:4, 0] = 1.0
    layer2[:, 4, 0] = 1.0
    layer1[:, 6:8, 0] = 1.0
    trajectory: dict[str, object] = {"hidden_spikes": (layer1, layer2)}
    tail, stages = exp02.tail_activity_terms(trajectory, lengths, fs)
    expected_stages = torch.tensor([0.25, 0.125, 0.25])
    assert torch.allclose(torch.stack(stages), expected_stages)
    expected = (1.0 * 0.25 + 2.0 * 0.125 + 4.0 * 0.25) / 7.0
    assert torch.allclose(tail, torch.tensor(expected))


def test_regularizer_profiles_have_expected_composition() -> None:
    p2 = torch.tensor(2.0)
    a1 = torch.tensor(3.0)
    tail = torch.tensor(4.0)
    assert torch.allclose(exp02.base_regularizer("a1", p2, a1, tail), torch.tensor(0.3))
    assert torch.allclose(exp02.base_regularizer("a1_p2", p2, a1, tail), torch.tensor(0.32))
    assert torch.allclose(exp02.base_regularizer("tail", p2, a1, tail), torch.tensor(0.4))
    assert torch.allclose(exp02.base_regularizer("a1_tail", p2, a1, tail), torch.tensor(0.7))


def test_task_loss_uses_valid_lengths_even_when_tail_rollout_is_present() -> None:
    source = inspect.getsource(exp02._task_and_reg)
    assert "endpoint_rollout_input" in source
    assert "objective_loss" in source
    assert "lengths," in source
    assert "tail_activity_terms" in source


def test_p2_and_a1_reuse_exp014_all_fixes() -> None:
    source = inspect.getsource(exp02.valid_regularization_terms)
    assert "exp014.regularization_terms" in source
    assert 'condition="all_fixes"' in source


def test_gradient_calibration_is_hidden_only_one_time_and_five_percent() -> None:
    assert exp02.TARGET_GRAD_RATIO == 0.05
    assert exp02.CALIBRATION_BATCHES == 5
    assert exp02.WARMUP_EPOCHS == 10
    calibration = inspect.getsource(exp02.calibrate_strength)
    training = inspect.getsource(exp02.train_one)
    assert "_hidden_parameters(model)" in calibration
    assert "exp015._gradient_pair_stats" in calibration
    assert "np.median" in calibration
    assert training.count("calibrate_strength(") == 1
    assert "kappa = float(calibration[\"kappa\"])" in training
    assert "kappa =" not in training.split("for epoch", 1)[1]


def test_frozen_loader_points_to_exp01_checkpoint_and_does_not_train() -> None:
    source = inspect.getsource(exp02._load_frozen_model)
    assert "exp01.checkpoint_path" in source
    assert "base_results_dir" in source
    assert "model_state_dict" in source
    frozen_eval = inspect.getsource(exp02.evaluate_frozen)
    assert "train_one(" not in frozen_eval


def test_finalizer_is_aggregation_only() -> None:
    source = inspect.getsource(exp02.finalize_experiment)
    for forbidden in ("train_one(", "run_one(", "evaluate_one(", "optimizer", ".backward("):
        assert forbidden not in source
    for artifact in (
        "summary.csv",
        "comparison_summary.csv",
        "paired_vs_frozen_exp01.csv",
        "calibration_summary.csv",
        "history_long.csv",
        "raster_index.csv",
        "manifest.json",
    ):
        assert artifact in source


def test_slurm_layout_matches_multi_cpu_contract() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    frozen_text = FROZEN_SCRIPT.read_text(encoding="utf-8")
    finalize_text = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-71%50" in array_text
    for text in (array_text, frozen_text, finalize_text):
        assert "#SBATCH --cpus-per-task=1" in text
        assert 'REPO_ROOT="${REPO_ROOT:-$PWD}"' in text
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert f"export {name}=1" in text

    assert 'REPO_ROOT="${REPO_ROOT:-$(pwd)}"' in submit_text
    assert "export REPO_ROOT" in submit_text
    assert "sbatch --parsable --export=ALL" in submit_text
    assert 'afterok:${ARRAY_JOB}:${FROZEN_JOB}' in submit_text


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
        "summary.csv",
        "comparison_summary.csv",
        "paired_vs_frozen_exp01.csv",
        "calibration_summary.csv",
        "history_long.csv",
        "raster_index.csv",
        "test_balanced_accuracy",
        "test_tail_area_final_hidden",
        "DISPLAY_SEED",
        "SAMPLE_INDEX",
    ):
        assert token in joined
    for forbidden in ("optimizer.step(", ".backward()", "subprocess", "sbatch", "run-one", "train_one", "multiprocessing"):
        assert forbidden not in joined
