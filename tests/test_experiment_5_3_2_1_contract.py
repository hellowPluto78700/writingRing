from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_5_3_2_1_when_objective_sweep as exp5321


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "scripts/bash_script/SNN_Bash/prepare_exp_5_3_2_1_local_cpu_array.bash"
RUNNER = REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_5_3_2_1_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_5_3_2_1_cpu.bash"
SUBMIT = REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_5_3_2_1_cpu.bash"
README = REPO_ROOT / "scripts/experiment_5_3_2_1/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_3_2_1_when_objective_sweep.ipynb"


def test_objective_matrix_is_five_by_five_with_fixed_rsnn() -> None:
    assert exp5321.PROTOCOL_VERSION == "rsnn_when_objective_v1"
    assert exp5321.SEEDS == (11, 23, 37, 53, 71)
    assert exp5321.WHEN_CONDITION == "rsnn_shortmem"
    assert exp5321.OBJECTIVE_NAMES == (
        "phase_only",
        "phase_heavy",
        "joint",
        "progress_heavy",
        "progress_only",
    )
    expected = {
        "phase_only": (1.0, 0.0),
        "phase_heavy": (5.0, 1.0),
        "joint": (1.0, 1.0),
        "progress_heavy": (0.2, 1.0),
        "progress_only": (0.0, 1.0),
    }
    for name, weights in expected.items():
        objective = exp5321.OBJECTIVE_BY_NAME[name]
        assert (objective.phase_weight, objective.progress_weight) == weights

    specs = exp5321.run_specs()
    assert len(specs) == 25
    assert len({spec.key for spec in specs}) == 25


def test_objective_loss_uses_declared_phase_and_progress_weights() -> None:
    phase = torch.tensor(2.0)
    progress = torch.tensor(0.25)
    expected = {
        "phase_only": 2.0,
        "phase_heavy": 10.25,
        "joint": 2.25,
        "progress_heavy": 0.65,
        "progress_only": 0.25,
    }
    for name, value in expected.items():
        loss = exp5321._objective_loss(
            exp5321.OBJECTIVE_BY_NAME[name],
            phase,
            progress,
        )
        assert torch.allclose(loss, torch.tensor(value))


def test_same_seed_uses_paired_rsnn_initialization_across_objectives() -> None:
    first = exp5321._initialize_model(11, 64.0, torch.device("cpu"))
    second = exp5321._initialize_model(11, 64.0, torch.device("cpu"))
    assert first.recurrent is not None
    assert second.recurrent is not None
    assert first.input_projection.in_features == 128
    assert first.input_projection.out_features == 128
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])


def test_gradient_diagnostic_tracks_weighted_objective_dominance() -> None:
    model = exp5321._initialize_model(23, 64.0, torch.device("cpu"))
    local = torch.randn(2, 5, 128)
    lengths = torch.tensor([5, 4])

    phase_only = exp5321.gradient_diagnostics(
        model,
        local,
        lengths,
        exp5321.OBJECTIVE_BY_NAME["phase_only"],
    )
    assert phase_only["weighted_phase_grad_fraction"] > 0.999
    assert phase_only["weighted_progress_grad_fraction"] < 1e-8

    progress_only = exp5321.gradient_diagnostics(
        model,
        local,
        lengths,
        exp5321.OBJECTIVE_BY_NAME["progress_only"],
    )
    assert progress_only["weighted_phase_grad_fraction"] < 1e-8
    assert progress_only["weighted_progress_grad_fraction"] > 0.999


def test_progress_trajectory_metrics_reward_forward_order() -> None:
    increasing = np.array([0.0, 0.2, 0.4, 0.7, 1.0])
    zigzag = np.array([0.0, 0.4, 0.2, 0.8, 0.6])
    assert exp5321._spearman_against_time(increasing) > 0.99
    assert exp5321._monotonic_violation_rate(increasing) == 0.0
    assert exp5321._spearman_against_time(zigzag) < 0.9
    assert exp5321._monotonic_violation_rate(zigzag) > 0.0


def test_multi_cpu_afterok_and_environment_contract() -> None:
    prep = PREP.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-4%5" in prep
    assert "#SBATCH --array=0-24%25" in runner
    for text in (prep, runner, finalizer):
        assert "#SBATCH --cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert f"export {name}=1" in text

    assert 'afterok:${LOCAL_JOB}' in submit
    assert 'afterok:${RUN_JOB}' in submit
    assert "25 runs" in submit


def test_notebook_is_analysis_only_and_reads_finalized_outputs() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp5321-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)

    for token in (
        "runs.csv",
        "histories.csv",
        "probe_runs.csv",
        "ablation_runs.csv",
        "history_gain_runs.csv",
        "gradient_diagnostics.csv",
        "baseline_runs.csv",
        "manifest.json",
        "phase_only",
        "phase_heavy",
        "joint",
        "progress_heavy",
        "progress_only",
        "elapsed_time_only",
        "spearman",
        "monotonic",
        "H_reset",
        "H_shuffle",
        "plt.subplots",
    ):
        assert token in joined

    for forbidden in (
        "optimizer.step(",
        ".backward()",
        "LogisticRegression(",
        "Ridge(",
        "subprocess",
        "sbatch",
        "run-one",
        "prepare-local",
    ):
        assert forbidden not in joined


def test_readme_states_objective_upper_bound_and_clock_control() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "RSNN WHEN objective upper-bound sweep",
        "phase_only",
        "phase_heavy",
        "joint",
        "progress_heavy",
        "progress_only",
        "elapsed_time_only",
        "Spearman",
        "monotonic violation rate",
        "H_{reset}",
        "H_{shuffle}",
        "gradient norms",
        "25 independent runs",
        "afterok",
        "analysis-only",
        "WHAT x WHEN fusion",
    ):
        assert token in text
