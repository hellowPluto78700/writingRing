from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_5_3_2_2_when_width_sweep as exp5322


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "scripts/bash_script/SNN_Bash/prepare_exp_5_3_2_2_local_cpu_array.bash"
RUNNER = REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_5_3_2_2_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_5_3_2_2_cpu.bash"
SUBMIT = REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_5_3_2_2_cpu.bash"
README = REPO_ROOT / "scripts/experiment_5_3_2_2/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_3_2_2_when_width_sweep.ipynb"


def test_width_matrix_is_five_by_five_and_width_major() -> None:
    assert exp5322.PROTOCOL_VERSION == "rsnn_when_width_v1"
    assert exp5322.SEEDS == (11, 23, 37, 53, 71)
    assert exp5322.WIDTHS == (16, 32, 64, 128, 256)
    assert exp5322.WIDTH_CONDITIONS == (
        "rsnn_h16",
        "rsnn_h32",
        "rsnn_h64",
        "rsnn_h128",
        "rsnn_h256",
    )
    assert exp5322.OBJECTIVE.name == "joint"
    assert (exp5322.OBJECTIVE.phase_weight, exp5322.OBJECTIVE.progress_weight) == (
        1.0,
        1.0,
    )

    specs = exp5322.run_specs()
    assert len(specs) == 25
    assert len({spec.key for spec in specs}) == 25
    assert [(spec.hidden_width, spec.seed) for spec in specs[:5]] == [
        (16, seed) for seed in exp5322.SEEDS
    ]
    assert [(spec.hidden_width, spec.seed) for spec in specs[10:15]] == [
        (64, seed) for seed in exp5322.SEEDS
    ]
    assert [(spec.hidden_width, spec.seed) for spec in specs[20:25]] == [
        (256, seed) for seed in exp5322.SEEDS
    ]


def test_architecture_changes_only_hidden_width() -> None:
    expected_decay = exp5322.parent.decay_from_shift(exp5322.SHORT_SHIFT)
    for width in exp5322.WIDTHS:
        model = exp5322.WidthWhenBranchNet(width, fs=64.0)
        assert model.input_projection.in_features == 128
        assert model.input_projection.out_features == width
        assert model.input_projection.bias is None
        assert model.recurrent.in_features == width
        assert model.recurrent.out_features == width
        assert model.recurrent.bias is None
        assert model.phase_head.in_features == width
        assert model.progress_head.in_features == width
        assert model.alpha_vector.shape == (width,)
        assert model.beta_vector.shape == (width,)
        assert torch.allclose(
            model.alpha_vector,
            torch.full((width,), expected_decay),
        )
        assert torch.allclose(
            model.beta_vector,
            torch.full((width,), expected_decay),
        )
        assert exp5322.THRESHOLD == exp5322.objective_parent.parent.THRESHOLD
        assert exp5322.RESET == exp5322.objective_parent.parent.RESET


def test_parameter_count_reports_quadratic_recurrent_growth() -> None:
    for width in exp5322.WIDTHS:
        model = exp5322.WidthWhenBranchNet(width, fs=64.0)
        counts = exp5322.parameter_counts(model)
        assert counts["input_projection"] == 128 * width
        assert counts["recurrent"] == width * width
        assert counts["phase_head"] == 10 * width + 10
        assert counts["progress_head"] == width + 1
        assert counts["trainable_total"] == width * width + 139 * width + 11


def test_initialization_is_deterministic_and_width_specific() -> None:
    first = exp5322._initialize_model(64, 23, 64.0, torch.device("cpu"))
    second = exp5322._initialize_model(64, 23, 64.0, torch.device("cpu"))
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])

    wider = exp5322._initialize_model(128, 23, 64.0, torch.device("cpu"))
    assert first.input_projection.weight.shape == (64, 128)
    assert wider.input_projection.weight.shape == (128, 128)
    assert not torch.equal(
        first.input_projection.weight,
        wider.input_projection.weight[:64],
    )


def test_loader_seed_is_width_independent_within_master_seed() -> None:
    for seed in exp5322.SEEDS:
        expected = exp5322.base.dseed(seed, "exp5_3_2", "train", "loader")
        assert exp5322._loader_seed(seed, "train") == expected


def test_joint_loss_is_sample_balanced_and_padding_invariant() -> None:
    model = exp5322._initialize_model(32, 11, 64.0, torch.device("cpu"))
    x = torch.randn(2, 7, 128)
    lengths = torch.tensor([3, 6])
    total, phase, progress, logits, progress_pred, trajectory = model.loss_components(
        x,
        lengths,
    )
    assert torch.allclose(total, phase + progress)
    assert logits.shape == (2, 7, 10)
    assert progress_pred.shape == (2, 7)
    assert trajectory.membranes.shape == (2, 7, 32)

    changed = x.clone()
    changed[0, 3:] = torch.randn_like(changed[0, 3:]) * 100.0
    changed[1, 6:] = torch.randn_like(changed[1, 6:]) * 100.0
    changed_total, changed_phase, changed_progress, *_ = model.loss_components(
        changed,
        lengths,
    )
    assert torch.allclose(total, changed_total)
    assert torch.allclose(phase, changed_phase)
    assert torch.allclose(progress, changed_progress)


def test_state_reset_removes_history_but_preserves_current_input() -> None:
    model = exp5322._initialize_model(16, 37, 64.0, torch.device("cpu"))
    x = torch.randn(2, 6, 128)
    lengths = torch.tensor([6, 6])
    ordered = model.forward_trajectory(x, lengths, reset_state_each_step=False)
    reset = model.forward_trajectory(x, lengths, reset_state_each_step=True)
    assert ordered.membranes.shape == reset.membranes.shape == (2, 6, 16)
    assert not torch.equal(ordered.membranes[:, 1:], reset.membranes[:, 1:])

    one_step = model.forward_trajectory(
        x[:, :1],
        torch.ones(2, dtype=torch.long),
        reset_state_each_step=True,
    )
    assert torch.allclose(reset.membranes[:, :1], one_step.membranes)


def test_activity_diagnostics_detect_dead_and_high_activity_neurons() -> None:
    spikes = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    metrics = exp5322._activity_diagnostics(spikes, fs=64.0)
    assert metrics["n_neurons"] == 3
    assert np.isclose(metrics["mean_firing_rate"], 0.25)
    assert np.isclose(metrics["mean_firing_rate_hz"], 16.0)
    assert np.isclose(metrics["dead_neuron_fraction"], 2.0 / 3.0)
    assert np.isclose(metrics["highly_active_neuron_fraction"], 1.0 / 3.0)


def test_effective_dimension_recovers_rank_one_and_full_rank_cases() -> None:
    t = np.linspace(-1.0, 1.0, 20)
    rank_one = np.stack([t, 2.0 * t, -3.0 * t], axis=1)
    rank_one_metrics = exp5322._representation_diagnostics(rank_one)
    assert np.isclose(rank_one_metrics["effective_dimension"], 1.0, atol=1e-6)
    assert rank_one_metrics["pca90_dimension"] == 1

    full = np.eye(4, dtype=np.float64)
    full_metrics = exp5322._representation_diagnostics(full)
    assert 2.9 < full_metrics["effective_dimension"] <= 3.1
    assert full_metrics["pca90_dimension"] == 3


def test_evaluation_artifact_contract_is_explicit() -> None:
    assert exp5322.EVALUATION_REQUIRED_SECTIONS == (
        "parameter_count",
        "parameter_counts",
        "native",
        "probes",
        "trajectory_metrics",
        "history_attribution",
        "activity",
        "representation",
    )


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


def test_notebook_is_analysis_only_and_reads_all_final_outputs() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp5322-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)

    for token in (
        "runs.csv",
        "histories.csv",
        "probe_runs.csv",
        "ablation_runs.csv",
        "history_gain_runs.csv",
        "activity_runs.csv",
        "representation_runs.csv",
        "baseline_runs.csv",
        "local_reference.csv",
        "manifest.json",
        "Capacity curve",
        "Generalization",
        "Temporal quality",
        "History contribution",
        "SNN-native accessibility",
        "Activity utilization",
        "Effective state dimension",
        "H_reset",
        "H_shuffle",
        "effective_dimension",
        "pca90_dimension",
        "elapsed_time_only",
        "what_only",
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


def test_readme_freezes_width_only_capacity_contract() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "RSNN hidden-state capacity sweep",
        "pure width sweep",
        "rsnn_h16",
        "rsnn_h32",
        "rsnn_h64",
        "rsnn_h128",
        "rsnn_h256",
        "25 independent runs",
        "L_{WHEN}=L_{phase}+L_{progress}",
        "`T_i` is used **only to construct supervision targets and the valid mask**",
        "dseed(seed, experiment, width, component)",
        "H_{reset}",
        "H_{shuffle}",
        "dead-neuron fraction",
        "highly-active-neuron fraction",
        "D_{eff}",
        "D_{90}",
        "activity_runs.csv",
        "representation_runs.csv",
        "25/25",
        "afterok",
        "analysis-only",
        "WHAT_{128}\\rightarrow FF\\text{-}SNN\\rightarrow RSNN",
    ):
        assert token in text
