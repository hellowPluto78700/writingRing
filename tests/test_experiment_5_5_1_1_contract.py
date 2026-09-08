from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_5_5_1_1_routing_diagnostics as exp5511
from scripts import experiment_5_5_1_regularized_semantic_when as exp551


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts/bash_script/SNN_Bash"
README = REPO_ROOT / "scripts/experiment_5_5_1_1/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_5_1_1_routing_diagnostics.ipynb"
RUNNER = "run_exp_5_5_1_1_diagnostic_cpu_array.bash"
FINALIZER = "finalize_exp_5_5_1_1_cpu.bash"
SUBMITTER = "submit_exp_5_5_1_1_cpu.bash"


def test_protocol_is_posthoc_and_seed_parallel() -> None:
    assert exp5511.PROTOCOL_VERSION == "routing_diagnostics_v1"
    assert exp5511.SEEDS == (11, 23, 37, 53, 71)
    assert exp5511.TEMPERATURES == (1.0, 0.8, 0.6, 0.4, 0.25)
    assert exp5511.HARD_VARIANT == "hard_top1"
    assert exp5511.SHUFFLE_REPLICATES == 5
    assert exp5511.variant_name(1.0) == "tau_1.00"
    assert exp5511.variant_name(0.25) == "tau_0.25"


def test_temperature_one_preserves_q_and_lower_temperature_sharpens() -> None:
    q = torch.tensor(
        [
            [
                [0.40, 0.30, 0.20, 0.10],
                [0.35, 0.30, 0.20, 0.15],
                [0.50, 0.20, 0.20, 0.10],
                [0.00, 0.00, 0.00, 0.00],
            ]
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([3])
    tau1 = exp5511.transform_q(q, lengths, temperature=1.0)
    tau05 = exp5511.transform_q(q, lengths, temperature=0.5)
    assert torch.allclose(tau1, q, atol=1e-7, rtol=1e-7)
    assert float(tau05[:, :3].amax(dim=-1).mean()) > float(tau1[:, :3].amax(dim=-1).mean())
    assert torch.equal(tau05[:, 3], torch.zeros_like(tau05[:, 3]))


def test_hard_top1_is_one_hot_on_valid_prefix_and_zero_on_padding() -> None:
    q = torch.tensor(
        [[[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.4, 0.5, 0.1], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([3])
    hard = exp5511.transform_q(q, lengths, hard_top1=True)
    assert torch.equal(hard[:, :3].sum(dim=-1), torch.ones(1, 3))
    assert torch.all((hard[:, :3] == 0) | (hard[:, :3] == 1))
    assert torch.equal(hard[:, 3], torch.zeros_like(hard[:, 3]))


def test_confidence_metrics_ignore_padding() -> None:
    q_a = np.asarray(
        [[[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    q_b = q_a.copy()
    q_b[:, 3] = np.asarray([0.99, 0.005, 0.005], dtype=np.float32)
    lengths = np.asarray([3], dtype=np.int64)
    assert exp5511.confidence_metrics(q_a, lengths) == exp5511.confidence_metrics(q_b, lengths)


def test_expert_weight_diagnostic_has_all_unordered_pairs() -> None:
    torch.manual_seed(7)
    model = exp551.RegularizedSemanticWhen(exp551.ORDERED)
    rows = exp5511.expert_weight_similarity_rows(model, seed=11)
    assert len(rows) == 28
    assert {(row["expert_a"], row["expert_b"]) for row in rows} == {
        (left, right) for left in range(8) for right in range(left + 1, 8)
    }


def test_identical_experts_have_zero_functional_difference() -> None:
    torch.manual_seed(9)
    model = exp551.RegularizedSemanticWhen(exp551.ORDERED)
    with torch.no_grad():
        template = model.expert_weight[0].clone()
        for expert in range(model.n_states):
            model.expert_weight[expert].copy_(template)
    trajectory = {
        "what": np.ones((2, 4, 128), dtype=np.uint8),
        "q": np.zeros((2, 4, 8), dtype=np.float32),
        "evidence": np.zeros((2, 4, 12), dtype=np.float32),
        "progress": np.zeros((2, 4), dtype=np.float32),
        "labels": np.asarray([0, 1], dtype=np.int64),
        "lengths": np.asarray([4, 3], dtype=np.int64),
    }
    rows = exp5511.expert_functional_diversity_rows(
        model,
        trajectory,
        seed=11,
        split="val",
    )
    assert len(rows) == 28
    for row in rows:
        assert abs(float(row["mean_evidence_l1_difference"])) < 1e-10
        assert abs(float(row["mean_evidence_l2_difference"])) < 1e-10
        assert abs(float(row["mean_relative_evidence_l2_difference"])) < 1e-10
        assert float(row["top_support_disagreement_rate"]) == 0.0


def test_runner_and_finalizer_are_diagnostic_only() -> None:
    runner_source = inspect.getsource(exp5511.run_seed)
    assert "backward(" not in runner_source
    assert "optimizer" not in runner_source.lower()
    assert "_train_model(" not in runner_source
    assert "train_screen_one(" not in runner_source
    assert "train_reset_one(" not in runner_source
    assert "no_temperature_selection" in runner_source
    finalizer_source = inspect.getsource(exp5511.finalize_experiment)
    assert "run_seed(" not in finalizer_source
    assert "Finalizer will not run missing diagnostic seed" in finalizer_source


def test_slurm_seed_array_and_dependency_contract() -> None:
    runner = (SLURM_DIR / RUNNER).read_text(encoding="utf-8")
    finalizer = (SLURM_DIR / FINALIZER).read_text(encoding="utf-8")
    submitter = (SLURM_DIR / SUBMITTER).read_text(encoding="utf-8")
    for text in (runner, finalizer):
        assert text.startswith("#!/usr/bin/env bash")
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
    assert "#SBATCH --array=0-4%5" in runner
    assert "run-seed" in runner
    assert "command -v sbatch" in submitter
    assert 'dependency="afterok:${DIAG_JOB}"' in submitter
    assert "--wrap" not in submitter
    assert "/usr/bin/sbatch" not in submitter


def test_notebook_and_readme_are_artifact_driven() -> None:
    assert README.is_file()
    readme = README.read_text(encoding="utf-8")
    assert "no training" in readme.lower()
    assert "5 independent seed diagnostic tasks" in readme
    assert "analysis-only" in readme
    assert NOTEBOOK.is_file()
    notebook_text = NOTEBOOK.read_text(encoding="utf-8")
    notebook = json.loads(notebook_text)
    assert notebook["nbformat"] == 4
    for artifact in (
        "confidence_metrics.csv",
        "confidence_summary.csv",
        "temperature_sweep.csv",
        "temperature_summary.csv",
        "temperature_alignment.csv",
        "expert_weight_similarity.csv",
        "expert_functional_diversity.csv",
        "expert_summary.csv",
        "manifest.json",
    ):
        assert artifact in notebook_text
    assert "analysis-only" in notebook_text.lower()
    assert "run_seed(" not in notebook_text
    assert "finalize_experiment(" not in notebook_text
    assert "sbatch" not in notebook_text
