from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_5_5_4_causal_when_phase_recovery as exp554


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts/bash_script/SNN_Bash"
README = REPO_ROOT / "scripts/experiment_5_5_4/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_5_4_causal_when_phase_recovery.ipynb"
RUNNERS = (
    "validate_exp_5_5_4_source_cpu_array.bash",
    "run_exp_5_5_4_when_cpu_array.bash",
    "finalize_exp_5_5_4_cpu.bash",
)
SUBMITTER = "submit_exp_5_5_4_cpu.bash"


def test_protocol_and_run_mapping() -> None:
    assert exp554.PROTOCOL_VERSION == "causal_when_phase_recovery_v1"
    assert exp554.SEEDS == (11, 23, 37, 53, 71)
    assert exp554.WHAT_WIDTH == 128
    assert exp554.STATE_WIDTH == 64
    assert exp554.N_PHASES == 10
    assert exp554.ARCHITECTURES == ("gru64_progress", "rsnn64_progress")
    specs = exp554.run_specs()
    assert len(specs) == 10
    assert len({spec.key for spec in specs}) == 10
    assert specs[0] == exp554.RunSpec(exp554.GRU, 11)
    assert specs[-1] == exp554.RunSpec(exp554.RSNN, 71)


def test_r10_progress_target_exactly_matches_relative10_phase_rule() -> None:
    lengths = torch.tensor([13, 10, 7], dtype=torch.long)
    target = exp554.r10_progress_targets(lengths, 13, torch.float64)
    phase_from_progress = torch.floor(target * exp554.N_PHASES).long().clamp(max=exp554.N_PHASES - 1)
    direct_phase = exp554.r10_phase_targets(lengths, 13)
    valid = exp554.sequence_mask(lengths, 13)
    assert torch.equal(phase_from_progress[valid], direct_phase[valid])


def test_progress_loss_is_sample_balanced() -> None:
    pred = torch.zeros((2, 4), dtype=torch.float32)
    lengths = torch.tensor([1, 4], dtype=torch.long)
    loss = exp554.sample_balanced_progress_loss(pred, lengths)
    target = exp554.r10_progress_targets(lengths, 4, pred.dtype)
    valid = exp554.sequence_mask(lengths, 4).to(pred.dtype)
    raw = torch.nn.functional.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=exp554.PROGRESS_BETA,
    )
    expected = ((raw * valid).sum(dim=1) / lengths.to(pred.dtype)).mean()
    assert torch.allclose(loss, expected)


def test_local2_uses_at_most_two_adjacent_experts_and_is_normalized() -> None:
    progress = np.asarray([[0.0, 0.05, 0.10, 0.34, 0.95, 1.0]], dtype=np.float64)
    lengths = np.asarray([6], dtype=np.int64)
    q = exp554.routing_probabilities_from_progress(progress, lengths, exp554.LOCAL2)
    assert np.allclose(q.sum(axis=2), 1.0)
    for timestep in range(6):
        active = np.flatnonzero(q[0, timestep] > 0.0)
        assert len(active) <= 2
        if len(active) == 2:
            assert active[1] - active[0] == 1


def test_hard_predicted_routing_matches_floor_10r() -> None:
    progress = np.asarray([[0.0, 0.099, 0.1, 0.299, 0.3, 0.999]], dtype=np.float64)
    lengths = np.asarray([6], dtype=np.int64)
    q = exp554.routing_probabilities_from_progress(progress, lengths, exp554.HARD)
    assert np.array_equal(q.argmax(axis=2)[0], np.asarray([0, 0, 1, 2, 3, 9]))


def test_training_source_contains_no_classification_loss_or_teacher_updates() -> None:
    source = inspect.getsource(exp554._train_progress_model)
    assert "cross_entropy" not in source
    assert "weight_bank" not in source
    assert "teacher_logits" not in source
    assert "sample_balanced_progress_loss" in source
    assert "_checkpoint_better" in source
    assert '"classification_gradient_used": False' in source


def test_checkpoint_selection_is_validation_when_only() -> None:
    source = inspect.getsource(exp554._checkpoint_better)
    assert "sample_balanced_progress_mae" in source
    assert "sample_balanced_smooth_l1" in source
    assert "progress_spearman" in source
    train_source = inspect.getsource(exp554._train_progress_model)
    assert 'eval_loaders["val"]' in train_source
    assert '"classification_gradient_used": False' in train_source
    assert '"test_used_for_checkpoint_selection": False' in train_source


def test_source_validation_refuses_to_retrain_teacher() -> None:
    source = inspect.getsource(exp554.validate_source_seed)
    assert "prepare_teacher_seed(" not in source
    assert "run_train_one(" not in source
    assert "LogisticRegression" not in source
    assert "never retrains the Relative10 teacher" in source
    assert "fatal oracle-hard gate failed" in source


def test_finalizer_is_artifact_only() -> None:
    source = inspect.getsource(exp554.finalize_experiment)
    assert "will not regenerate" in source
    assert "will not retrain or reevaluate" in source
    assert "run_when_one(" not in source
    assert "validate_source_seed(" not in source
    assert "_train_progress_model(" not in source


def test_slurm_multi_cpu_dependency_contract() -> None:
    paths = {name: SLURM_DIR / name for name in RUNNERS + (SUBMITTER,)}
    for name, path in paths.items():
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        assert "--wrap" not in text
        assert "/usr/bin/sbatch" not in text

    source = paths[RUNNERS[0]].read_text(encoding="utf-8")
    when = paths[RUNNERS[1]].read_text(encoding="utf-8")
    finalize = paths[RUNNERS[2]].read_text(encoding="utf-8")
    submit = paths[SUBMITTER].read_text(encoding="utf-8")

    assert "#SBATCH --array=0-4%5" in source
    assert "#SBATCH --array=0-9%10" in when
    for text in (source, when, finalize):
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
    assert 'dependency="afterok:${WHEN_JOB}"' in submit


def test_readme_and_notebook_are_analysis_artifact_driven() -> None:
    assert README.is_file()
    readme = README.read_text(encoding="utf-8")
    assert "10 WHEN train -> best checkpoint -> evaluate tasks" in readme
    assert "classification CE" in readme
    assert "artifact-only finalizer" in readme
    assert "analysis-only" in readme

    assert NOTEBOOK.is_file()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    text = NOTEBOOK.read_text(encoding="utf-8")
    for artifact in (
        "runs.csv",
        "summary.csv",
        "paired_deltas.csv",
        "phase_metrics.csv",
        "phase_confusion.csv",
        "routing_disagreement.csv",
        "margin_degradation.csv",
        "phase_sensitivity.csv",
    ):
        assert artifact in text
    for forbidden in (
        "run_when_one",
        "validate_source_seed",
        "subprocess",
        "sbatch",
        "torch.optim",
        "cross_entropy",
    ):
        assert forbidden not in text
