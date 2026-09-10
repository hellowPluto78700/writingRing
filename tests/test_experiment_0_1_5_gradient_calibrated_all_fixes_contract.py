from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import experiment_0_1_5_gradient_calibrated_all_fixes as exp015


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_0_1_5_gradient_calibrated_all_fixes.ipynb"
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_0_1_5_gradient_calibrated_all_fixes_cpu_array.bash"
FINALIZE_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_0_1_5_gradient_calibrated_all_fixes_cpu.bash"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_0_1_5_gradient_calibrated_all_fixes_cpu.bash"


def test_run_matrix_is_exactly_90_binary_gradient_calibrated_runs() -> None:
    specs = exp015.run_specs()
    assert len(specs) == 90
    assert len({spec.key for spec in specs}) == 90
    assert exp015.TARGET_GRAD_RATIOS == (0.05, 0.10, 0.20)
    assert exp015.SEEDS == (11, 23, 37, 53, 71)
    assert exp015.OBJECTIVES == ("whole_count_ce", "timestep_ce")
    assert {spec.architecture for spec in specs} == set(exp015.DIRECT_ARCHITECTURES)
    assert {spec.target_grad_ratio for spec in specs} == set(exp015.TARGET_GRAD_RATIOS)
    assert all("allfix_gr" in spec.key for spec in specs)
    assert exp015.VARIANT == "binary"
    assert exp015.HIDDEN_CAP == 1
    assert exp015.OUTPUT_CAP == 1


def test_training_and_early_stop_contract() -> None:
    assert exp015.CALIBRATION_BATCHES == 5
    assert exp015.WARMUP_EPOCHS == 10
    assert exp015.MIN_EPOCHS == 50
    assert exp015.MAX_EPOCHS == 100
    assert exp015.EARLY_STOP_PATIENCE == 30
    assert exp015.VAL_OBJECTIVE_LOSS_MIN_DELTA == 1e-4
    assert exp015._early_stop_triggered(49, 1) is False
    assert exp015._early_stop_triggered(50, 20) is True
    assert exp015._early_stop_triggered(50, 21) is False
    assert exp015._early_stop_triggered(75, 45) is True
    assert exp015._early_stop_triggered(75, 46) is False


def test_kappa_calibration_is_target_over_observed_ratio() -> None:
    assert exp015.calibrated_kappa(0.10, 0.02) == 5.0
    assert exp015.calibrated_kappa(0.05, 0.50) == 0.1
    assert exp015.calibrated_kappa(0.20, 0.10) == 2.0


def test_all_fixes_terms_delegate_to_exp014_all_fixes() -> None:
    spikes = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    pre = torch.tensor([[[1.0, 2.0], [2.0, 4.0]]])
    trajectory: dict[str, object] = {
        "hidden_spikes": (spikes,),
        "hidden_pre_reset": (pre,),
    }
    lengths = torch.tensor([2])
    alphas = (torch.tensor([0.5, 0.75]),)
    got = exp015.all_fixes_terms(trajectory, lengths, alphas)
    expected = exp015.exp014.regularization_terms(
        trajectory,
        lengths,
        alphas,
        condition="all_fixes",
        tau=exp015.REG_TAU,
    )
    assert torch.allclose(got[0], expected[0])
    assert torch.allclose(got[1], expected[1])
    assert len(got[2]) == len(expected[2]) == 1
    assert torch.allclose(got[2][0], expected[2][0])


def test_gradient_calibration_is_hidden_only_and_one_time() -> None:
    calibration_source = inspect.getsource(exp015.calibrate_strength)
    training_source = inspect.getsource(exp015.train_one)
    assert "_hidden_parameters(model)" in calibration_source
    assert "CALIBRATION_BATCHES" in calibration_source
    assert "np.median" in calibration_source
    assert "calibrated_kappa" in calibration_source
    assert training_source.count("calibrate_strength(") == 1
    assert "kappa = float(calibration[\"kappa\"])" in training_source
    assert "kappa =" not in training_source.split("for epoch", 1)[1]


def test_exp01_architecture_and_optimizer_contract_is_preserved() -> None:
    assert exp015.DIRECT_ARCHITECTURES == exp015.exp01.DIRECT_ARCHITECTURES
    assert exp015.OBJECTIVES == exp015.exp01.OBJECTIVES
    assert exp015.SEEDS == exp015.exp01.SEEDS
    assert exp015.HIDDEN_WIDTH == exp015.exp01.HIDDEN_WIDTH
    assert exp015.BATCH_SIZE == exp015.exp01.BATCH_SIZE
    assert exp015.LR == exp015.exp01.LR
    assert exp015.WEIGHT_DECAY == exp015.exp01.WEIGHT_DECAY


def test_finalizer_is_aggregation_only_and_reuses_frozen_exp01() -> None:
    source = inspect.getsource(exp015.finalize_experiment)
    for forbidden in ("train_one(", "run_one(", "optimizer", ".backward("):
        assert forbidden not in source
    assert "_load_frozen_exp01_binary_runs" in source
    assert "paired_vs_frozen_exp01.csv" in source
    assert "gradient_trajectory_summary.csv" in source
    assert "stopping_summary.csv" in source
    assert "calibration_summary.csv" in source


def test_slurm_follows_repository_multi_cpu_contract() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    finalize_text = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-89%50" in array_text
    assert "#SBATCH --cpus-per-task=1" in array_text
    assert "#SBATCH --cpus-per-task=1" in finalize_text
    assert "#SBATCH --time=24:00:00" in array_text
    for text in (array_text, finalize_text):
        assert 'REPO_ROOT="${REPO_ROOT:-$PWD}"' in text
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert f"export {name}=1" in text
        assert "outputs/" not in text
        assert "mkdir -p outputs" not in text

    assert 'REPO_ROOT="${REPO_ROOT:-$(pwd)}"' in submit_text
    assert "export REPO_ROOT" in submit_text
    assert "sbatch --parsable --export=ALL" in submit_text
    assert 'afterok:${ARRAY_JOB}' in submit_text
    assert "outputs/" not in submit_text


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
        "paired_vs_frozen_exp01_summary.csv",
        "calibration_summary.csv",
        "gradient_trajectory_summary.csv",
        "stopping_summary.csv",
        "history_long.csv",
        "comparison_summary.csv",
        "target_grad_ratio",
        "delta_test_ba_vs_frozen_exp01",
        "first_batch_effective_reg_to_task_hidden_grad_ratio",
        "first_batch_task_reg_hidden_grad_cosine",
    ):
        assert token in joined

    for forbidden in (
        "optimizer.step(",
        ".backward()",
        "subprocess",
        "sbatch",
        "run-one",
        "train_one",
        "multiprocessing",
    ):
        assert forbidden not in joined
