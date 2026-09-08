from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import experiment_5_5_non_snn_history_experts as exp55


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts/bash_script/SNN_Bash"
README = REPO_ROOT / "scripts/experiment_5_5/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_5_non_snn_history_experts.ipynb"
RUNNERS = (
    "prepare_exp_5_5_reference_cpu_array.bash",
    "run_exp_5_5_cpu_array.bash",
    "finalize_exp_5_5_cpu.bash",
)
SUBMITTER = "submit_exp_5_5_cpu.bash"


def test_exp55_protocol_and_array_mapping() -> None:
    assert exp55.PROTOCOL_VERSION == "non_snn_history_experts_v1"
    assert exp55.SEEDS == (11, 23, 37, 53, 71)
    assert exp55.CONDITIONS == (
        "shared",
        "clock",
        "oracle_progress",
        "reset_gru",
        "ordered_gru",
    )
    assert exp55.GRU_WIDTH == 64
    assert exp55.N_STATES == 8
    specs = exp55.run_specs()
    assert len(specs) == 25
    assert len({spec.key for spec in specs}) == 25
    assert specs[0] == exp55.RunSpec("shared", 11)
    assert specs[-1] == exp55.RunSpec("ordered_gru", 71)


def test_full_expert_and_matched_gru_parameter_contract() -> None:
    reset = exp55.NonSNNHistoryExperts(exp55.RESET_GRU)
    ordered = exp55.NonSNNHistoryExperts(exp55.ORDERED_GRU)
    clock = exp55.NonSNNHistoryExperts(exp55.CLOCK)
    shared = exp55.NonSNNHistoryExperts(exp55.SHARED)
    assert tuple(reset.expert_weight.shape) == (8, 12, 128)
    assert tuple(clock.expert_weight.shape) == (8, 12, 128)
    assert tuple(shared.shared_weight.shape) == (12, 128)
    assert exp55.parameter_counts(reset) == exp55.parameter_counts(ordered)
    assert reset.class_bias.shape == (12,)
    assert not hasattr(reset, "expert_bias")


def test_ordered_q_is_strict_history_not_current_what() -> None:
    torch.manual_seed(7)
    model = exp55.NonSNNHistoryExperts(exp55.ORDERED_GRU).eval()
    what_a = torch.randn(2, 7, 128)
    what_b = what_a.clone()
    what_b[:, 3, :] += 9.0
    lengths = torch.tensor([7, 6])
    with torch.no_grad():
        q_a = model.context_probabilities(what_a, lengths, 64.0, 2.0)
        q_b = model.context_probabilities(what_b, lengths, 64.0, 2.0)
    # q_t uses h_(t-1), so changing m_t must not change q_t.
    assert torch.equal(q_a[:, 3], q_b[:, 3])
    # The changed current WHAT is allowed to affect later state.
    assert not torch.equal(q_a[:, 4], q_b[:, 4])


def test_reset_q_has_no_cross_timestep_history() -> None:
    torch.manual_seed(11)
    model = exp55.NonSNNHistoryExperts(exp55.RESET_GRU).eval()
    what_a = torch.randn(2, 7, 128)
    what_b = what_a.clone()
    what_b[:, :3, :] += 6.0
    lengths = torch.tensor([7, 7])
    with torch.no_grad():
        q_a = model.context_probabilities(what_a, lengths, 64.0, 2.0)
        q_b = model.context_probabilities(what_b, lengths, 64.0, 2.0)
    # Current m_3 is identical and RESET_GRU has no state from m_0..m_2.
    assert torch.equal(q_a[:, 3], q_b[:, 3])


def test_padding_cannot_change_logits() -> None:
    torch.manual_seed(13)
    model = exp55.NonSNNHistoryExperts(exp55.ORDERED_GRU).eval()
    what_a = torch.randn(2, 9, 128)
    what_b = what_a.clone()
    lengths = torch.tensor([5, 7])
    what_b[0, 5:, :] = torch.randn_like(what_b[0, 5:, :]) * 100.0
    what_b[1, 7:, :] = torch.randn_like(what_b[1, 7:, :]) * 100.0
    with torch.no_grad():
        logits_a, _ = model(what_a, lengths, 64.0, 2.0)
        logits_b, _ = model(what_b, lengths, 64.0, 2.0)
    assert torch.equal(logits_a, logits_b)


def test_validation_checkpoint_selection_does_not_use_test() -> None:
    source = inspect.getsource(exp55.train_one)
    assert '("train", "val")' in source
    assert '"test"' not in source
    evaluation = inspect.getsource(exp55.evaluate_run_one)
    assert '"test"' in evaluation
    assert "load_model" in evaluation
    assert '"test_used_for_checkpoint_selection": False' in evaluation


def test_slurm_multi_cpu_contract() -> None:
    paths = {name: SLURM_DIR / name for name in RUNNERS + (SUBMITTER,)}
    for name, path in paths.items():
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        assert "--wrap" not in text
        assert "/usr/bin/sbatch" not in text
    prep = paths[RUNNERS[0]].read_text(encoding="utf-8")
    runs = paths[RUNNERS[1]].read_text(encoding="utf-8")
    finalize = paths[RUNNERS[2]].read_text(encoding="utf-8")
    submit = paths[SUBMITTER].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-4%5" in prep
    assert "#SBATCH --array=0-24%25" in runs
    assert "--cpus-per-task=1" in prep
    assert "--cpus-per-task=1" in runs
    assert "--cpus-per-task=1" in finalize
    for text in (prep, runs, finalize):
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
    assert 'dependency="afterok:${PREP_JOB}"' in submit
    assert 'dependency="afterok:${RUN_JOB}"' in submit


def test_finalizer_and_notebook_are_artifact_only() -> None:
    finalizer_source = inspect.getsource(exp55.finalize_experiment)
    assert "run_one(" not in finalizer_source
    assert "train_one(" not in finalizer_source
    assert "Finalizer will not retrain missing run" in finalizer_source
    assert README.is_file()
    readme = README.read_text(encoding="utf-8")
    assert "25 independent condition x seed tasks" in readme
    assert "analysis-only" in readme
    assert NOTEBOOK.is_file()
    notebook_text = NOTEBOOK.read_text(encoding="utf-8")
    notebook = json.loads(notebook_text)
    assert notebook["nbformat"] == 4
    assert "analysis-only" in notebook_text.lower()
    assert "runs.csv" in notebook_text
    assert "paired_deltas.csv" in notebook_text
    assert "state_profiles.csv" in notebook_text
    assert "q_only_probes.csv" in notebook_text
    assert "same_what_pairs.csv" in notebook_text
    assert "train_one(" not in notebook_text
    assert "run_one(" not in notebook_text
    assert "sbatch" not in notebook_text
