from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import experiment_5_5_non_snn_history_experts as exp55
from scripts import experiment_5_5_1_regularized_semantic_when as exp551


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts/bash_script/SNN_Bash"
README = REPO_ROOT / "scripts/experiment_5_5_1/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_5_1_regularized_semantic_when.ipynb"
RUNNERS = (
    "validate_exp_5_5_1_source_cpu_array.bash",
    "run_exp_5_5_1_screen_cpu_array.bash",
    "select_exp_5_5_1_screen_cpu.bash",
    "run_exp_5_5_1_reset_cpu_array.bash",
    "run_exp_5_5_1_final_cpu_array.bash",
    "finalize_exp_5_5_1_cpu.bash",
)
SUBMITTER = "submit_exp_5_5_1_cpu.bash"


def _has_grad(parameter: torch.Tensor) -> bool:
    return parameter.grad is not None and bool(torch.any(parameter.grad != 0))


def test_protocol_and_screen_mapping() -> None:
    assert exp551.PROTOCOL_VERSION == "regularized_semantic_when_v1"
    assert exp551.SEEDS == (11, 23, 37, 53, 71)
    assert exp551.GRU_WIDTH == 64
    assert exp551.N_STATES == 8
    assert exp551.WHAT_WIDTH == 128
    assert exp551.SCREEN_RECIPES == (
        "progress",
        "progress_sticky",
        "progress_sticky_confidence",
    )
    assert exp551.RECIPES[exp551.R1].lambda_progress == 0.3
    assert exp551.RECIPES[exp551.R2].lambda_sticky == 0.1
    assert exp551.RECIPES[exp551.R3].lambda_confidence == 0.02
    specs = exp551.screen_specs()
    assert len(specs) == 15
    assert len({spec.key for spec in specs}) == 15
    assert specs[0] == exp551.ScreenSpec(exp551.R1, 11)
    assert specs[-1] == exp551.ScreenSpec(exp551.R3, 71)


def test_exp55_core_initialization_is_preserved_before_progress_head() -> None:
    model_seed = exp551.base.dseed(11, exp55.EXPERIMENT_ID, "paired_constructor")
    exp551.base.seed_all(model_seed)
    source = exp55.NonSNNHistoryExperts(exp55.ORDERED_GRU)
    exp551.base.seed_all(model_seed)
    target = exp551.RegularizedSemanticWhen(exp551.ORDERED)
    assert torch.equal(source.class_bias, target.class_bias)
    assert torch.equal(source.expert_weight, target.expert_weight)
    for source_parameter, target_parameter in zip(source.gru.parameters(), target.gru.parameters()):
        assert torch.equal(source_parameter, target_parameter)
    for source_parameter, target_parameter in zip(source.state_head.parameters(), target.state_head.parameters()):
        assert torch.equal(source_parameter, target_parameter)


def test_progress_head_cannot_change_routing_or_logits() -> None:
    torch.manual_seed(3)
    model = exp551.RegularizedSemanticWhen(exp551.ORDERED).eval()
    what = torch.randn(2, 8, 128)
    lengths = torch.tensor([8, 6])
    with torch.no_grad():
        logits_a, trajectory_a = model(what, lengths, return_trajectory=True)
        assert trajectory_a is not None
        model.progress_head.weight.add_(100.0)
        model.progress_head.bias.add_(100.0)
        logits_b, trajectory_b = model(what, lengths, return_trajectory=True)
        assert trajectory_b is not None
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(trajectory_a.q, trajectory_b.q)
    assert not torch.equal(trajectory_a.progress, trajectory_b.progress)


def test_ordered_q_is_strict_history_while_progress_can_use_current_what() -> None:
    torch.manual_seed(7)
    model = exp551.RegularizedSemanticWhen(exp551.ORDERED).eval()
    what_a = torch.randn(2, 7, 128)
    what_b = what_a.clone()
    what_b[:, 3, :] += 9.0
    lengths = torch.tensor([7, 6])
    with torch.no_grad():
        _, trajectory_a = model(what_a, lengths, return_trajectory=True)
        _, trajectory_b = model(what_b, lengths, return_trajectory=True)
        assert trajectory_a is not None and trajectory_b is not None
    assert torch.equal(trajectory_a.q[:, 3], trajectory_b.q[:, 3])
    assert not torch.equal(trajectory_a.q[:, 4], trajectory_b.q[:, 4])
    assert not torch.equal(trajectory_a.progress[:, 3], trajectory_b.progress[:, 3])


def test_reset_has_no_cross_timestep_history() -> None:
    torch.manual_seed(11)
    model = exp551.RegularizedSemanticWhen(exp551.RESET).eval()
    what_a = torch.randn(2, 7, 128)
    what_b = what_a.clone()
    what_b[:, :3, :] += 6.0
    lengths = torch.tensor([7, 7])
    with torch.no_grad():
        _, trajectory_a = model(what_a, lengths, return_trajectory=True)
        _, trajectory_b = model(what_b, lengths, return_trajectory=True)
        assert trajectory_a is not None and trajectory_b is not None
    assert torch.equal(trajectory_a.q[:, 3], trajectory_b.q[:, 3])
    assert torch.equal(trajectory_a.progress[:, 3], trajectory_b.progress[:, 3])


def test_padding_cannot_change_logits() -> None:
    torch.manual_seed(13)
    model = exp551.RegularizedSemanticWhen(exp551.ORDERED).eval()
    what_a = torch.randn(2, 9, 128)
    what_b = what_a.clone()
    lengths = torch.tensor([5, 7])
    what_b[0, 5:, :] = torch.randn_like(what_b[0, 5:, :]) * 100.0
    what_b[1, 7:, :] = torch.randn_like(what_b[1, 7:, :]) * 100.0
    with torch.no_grad():
        logits_a, _ = model(what_a, lengths)
        logits_b, _ = model(what_b, lengths)
    assert torch.equal(logits_a, logits_b)


def test_auxiliary_losses_ignore_padding_and_invalid_transition_pairs() -> None:
    lengths = torch.tensor([3])
    logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0])
    q_a = torch.tensor(
        [[[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.5, 0.5]]],
        dtype=torch.float32,
    )
    q_b = q_a.clone()
    q_b[:, 3:, :] = torch.tensor([0.99, 0.01])
    progress_a = torch.tensor([[0.0, 0.5, 1.0, 0.2, 0.2]])
    progress_b = progress_a.clone()
    progress_b[:, 3:] = 0.99
    evidence = torch.zeros(1, 5, 2)
    recipe = exp551.Recipe("test", 1.0, 1.0, 1.0, 0)
    a = exp551.loss_components(
        logits,
        labels,
        exp551.Trajectory(q=q_a, evidence=evidence, progress=progress_a),
        lengths,
        recipe,
    )
    b = exp551.loss_components(
        logits,
        labels,
        exp551.Trajectory(q=q_b, evidence=evidence, progress=progress_b),
        lengths,
        recipe,
    )
    for name in ("progress", "sticky", "confidence"):
        assert torch.equal(a[name], b[name])


def test_progress_gradient_routes_only_to_gru_and_progress_head() -> None:
    torch.manual_seed(17)
    model = exp551.RegularizedSemanticWhen(exp551.ORDERED)
    what = torch.randn(3, 6, 128)
    labels = torch.tensor([0, 1, 2])
    lengths = torch.tensor([6, 5, 4])
    logits, trajectory = model(what, lengths, return_trajectory=True)
    assert trajectory is not None
    components = exp551.loss_components(logits, labels, trajectory, lengths, exp551.RECIPES[exp551.R1])
    components["progress"].backward()
    assert any(_has_grad(parameter) for parameter in model.gru.parameters())
    assert any(_has_grad(parameter) for parameter in model.progress_head.parameters())
    assert not _has_grad(model.expert_weight)
    assert not _has_grad(model.class_bias)
    assert not any(_has_grad(parameter) for parameter in model.state_head.parameters())


def test_state_regularizers_do_not_directly_train_experts_or_progress_head() -> None:
    torch.manual_seed(19)
    model = exp551.RegularizedSemanticWhen(exp551.ORDERED)
    what = torch.randn(3, 6, 128)
    labels = torch.tensor([0, 1, 2])
    lengths = torch.tensor([6, 5, 4])
    logits, trajectory = model(what, lengths, return_trajectory=True)
    assert trajectory is not None
    components = exp551.loss_components(logits, labels, trajectory, lengths, exp551.RECIPES[exp551.R3])
    (components["sticky"] + components["confidence"]).backward()
    assert any(_has_grad(parameter) for parameter in model.gru.parameters())
    assert any(_has_grad(parameter) for parameter in model.state_head.parameters())
    assert not _has_grad(model.expert_weight)
    assert not _has_grad(model.class_bias)
    assert not any(_has_grad(parameter) for parameter in model.progress_head.parameters())


def test_classification_ce_trains_routing_and_experts_but_not_progress_head() -> None:
    torch.manual_seed(23)
    model = exp551.RegularizedSemanticWhen(exp551.ORDERED)
    what = torch.randn(3, 6, 128)
    labels = torch.tensor([0, 1, 2])
    lengths = torch.tensor([6, 5, 4])
    logits, trajectory = model(what, lengths, return_trajectory=True)
    assert trajectory is not None
    components = exp551.loss_components(logits, labels, trajectory, lengths, exp551.RECIPES[exp551.R1])
    components["classification"].backward()
    assert _has_grad(model.expert_weight)
    assert _has_grad(model.class_bias)
    assert any(_has_grad(parameter) for parameter in model.gru.parameters())
    assert any(_has_grad(parameter) for parameter in model.state_head.parameters())
    assert not any(_has_grad(parameter) for parameter in model.progress_head.parameters())


def test_screen_training_and_evaluation_do_not_construct_test_loader() -> None:
    train_source = inspect.getsource(exp551._train_model)
    screen_source = inspect.getsource(exp551.evaluate_screen_one)
    assert '("train", "val")' in train_source
    assert '("train", "val")' in screen_source
    assert '("train", "val", "test")' not in train_source
    assert '("train", "val", "test")' not in screen_source
    assert '"test_evaluated": False' in screen_source
    final_source = inspect.getsource(exp551.evaluate_final_one)
    assert '("train", "val", "test")' in final_source
    assert "load_selection" in final_source


def test_finalizer_is_artifact_only() -> None:
    source = inspect.getsource(exp551.finalize_experiment)
    assert "train_screen_one(" not in source
    assert "train_reset_one(" not in source
    assert "evaluate_final_one(" not in source
    assert "Finalizer will not" in source


def test_slurm_multi_cpu_dependency_contract() -> None:
    paths = {name: SLURM_DIR / name for name in RUNNERS + (SUBMITTER,)}
    for name, path in paths.items():
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        assert "--wrap" not in text
        assert "/usr/bin/sbatch" not in text
    source = paths[RUNNERS[0]].read_text(encoding="utf-8")
    screen = paths[RUNNERS[1]].read_text(encoding="utf-8")
    select = paths[RUNNERS[2]].read_text(encoding="utf-8")
    reset = paths[RUNNERS[3]].read_text(encoding="utf-8")
    final = paths[RUNNERS[4]].read_text(encoding="utf-8")
    finalize = paths[RUNNERS[5]].read_text(encoding="utf-8")
    submit = paths[SUBMITTER].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-4%5" in source
    assert "#SBATCH --array=0-14%15" in screen
    assert "#SBATCH --array=0-4%5" in reset
    assert "#SBATCH --array=0-4%5" in final
    for text in (source, screen, select, reset, final, finalize):
        assert "--cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert 'eval "$(conda shell.bash hook)"' in text
        assert "conda activate writingring-gpu" in text
        assert "conda activate writingring-viz" in text
        for variable in (
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
        ):
            assert variable in text
    assert "command -v sbatch" in submit
    assert 'dependency="afterok:${SOURCE_JOB}"' in submit
    assert 'dependency="afterok:${SCREEN_JOB}"' in submit
    assert 'dependency="afterok:${SELECT_JOB}"' in submit
    assert 'dependency="afterok:${RESET_JOB}"' in submit
    assert 'dependency="afterok:${FINAL_JOB}"' in submit


def test_readme_and_notebook_are_analysis_artifact_driven() -> None:
    assert README.is_file()
    readme = README.read_text(encoding="utf-8")
    assert "15 ordered screen tasks" in readme
    assert "5 matched reset tasks" in readme
    assert "analysis-only" in readme
    assert NOTEBOOK.is_file()
    notebook_text = NOTEBOOK.read_text(encoding="utf-8")
    notebook = json.loads(notebook_text)
    assert notebook["nbformat"] == 4
    for artifact in (
        "screen_summary.csv",
        "selection.json",
        "final_runs.csv",
        "final_paired_deltas.csv",
        "progress_metrics.csv",
        "state_diagnostics.csv",
        "state_profiles.csv",
        "q_only_probes.csv",
        "q_progress_probe.csv",
        "matched_progress_same_what.csv",
    ):
        assert artifact in notebook_text
    assert "analysis-only" in notebook_text.lower()
    assert "train_screen_one(" not in notebook_text
    assert "train_reset_one(" not in notebook_text
    assert "evaluate_final_one(" not in notebook_text
    assert "select_screen(" not in notebook_text
    assert "sbatch" not in notebook_text
