from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_7_1_long_tau_antipersistence as exp71


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_grid_is_three_conditions_by_three_paired_seeds() -> None:
    specs = exp71.run_specs()
    assert len(specs) == 9
    assert {spec.condition for spec in specs} == set(exp71.CONDITIONS)
    assert {spec.seed for spec in specs} == {11, 23, 37}
    for seed in exp71.TRAIN_SEEDS:
        assert len([spec for spec in specs if spec.seed == seed]) == 3


def test_hierarchical_timescales_and_training_contract() -> None:
    assert exp71.HIERARCHICAL_SHIFTS == (4, 5, 6)
    taus = [exp71.exp70.tau_ms_from_shift(shift, 64.0) for shift in exp71.HIERARCHICAL_SHIFTS]
    assert np.allclose(taus, [242.10347130039656, 492.1461612435418, 992.1669944116244])
    assert exp71.HIDDEN_WIDTH == 128
    assert exp71.HIDDEN_CAP == 1
    assert exp71.OUTPUT_CAP == 1
    assert exp71.MAX_EPOCHS == 100
    assert exp71.MIN_EPOCHS == 20
    assert exp71.PATIENCE == 30
    assert exp71.WARMUP_EPOCHS == 10


def test_backbone_is_strict_serial_and_output_reads_long_only() -> None:
    model = exp71.SerialLongTauSNN(n_classes=12, fs=64.0)
    assert model.short_linear.in_features == 30
    assert model.short_linear.out_features == 128
    assert model.middle_linear.in_features == 128
    assert model.middle_linear.out_features == 128
    assert model.long_linear.in_features == 128
    assert model.long_linear.out_features == 128
    assert model.output_linear.in_features == 128
    assert model.output_linear.out_features == 12
    assert not hasattr(model, "short_head")
    assert not hasattr(model, "middle_head")
    assert not hasattr(model, "gain_head")


def test_rate_mask_policy_matches_protocol() -> None:
    spikes = torch.tensor([[[1.0], [0.0], [1.0], [1.0]]])
    lengths = torch.tensor([2])
    masked = exp71.long_rate_loss(spikes, lengths, exp71.ALL_LOSS_MASKED)
    full = exp71.long_rate_loss(spikes, lengths, exp71.WHOLECOUNT_ONLY_MASKED)
    assert torch.isclose(masked, torch.tensor(0.5))
    assert torch.isclose(full, torch.tensor(0.75))


def test_persistence_penalty_detects_continuous_and_high_duty_firing() -> None:
    lengths3 = torch.tensor([3])
    all_ones = torch.ones(1, 3, 1)
    p111 = exp71.persistence_loss(
        all_ones,
        lengths3,
        exp71.WHOLECOUNT_ONLY_MASKED,
        window=3,
        allowed=2,
    )
    assert torch.isclose(p111, torch.tensor(1.0))

    allowed_pattern = torch.tensor([[[1.0], [1.0], [0.0]]])
    p110 = exp71.persistence_loss(
        allowed_pattern,
        lengths3,
        exp71.WHOLECOUNT_ONLY_MASKED,
        window=3,
        allowed=2,
    )
    assert torch.isclose(p110, torch.tensor(0.0))

    high_duty = torch.tensor(
        [[[1.0], [1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0]]]
    )
    p84 = exp71.persistence_loss(
        high_duty,
        torch.tensor([8]),
        exp71.WHOLECOUNT_ONLY_MASKED,
        window=8,
        allowed=4,
    )
    assert torch.isclose(p84, torch.tensor(1.0))


def test_valid_mask_requires_entire_persistence_window_inside_valid_region() -> None:
    spikes = torch.ones(1, 3, 1)
    masked = exp71.persistence_loss(
        spikes,
        torch.tensor([2]),
        exp71.ALL_LOSS_MASKED,
        window=3,
        allowed=2,
    )
    full = exp71.persistence_loss(
        spikes,
        torch.tensor([2]),
        exp71.WHOLECOUNT_ONLY_MASKED,
        window=3,
        allowed=2,
    )
    assert torch.isclose(masked, torch.tensor(0.0))
    assert torch.isclose(full, torch.tensor(1.0))


def test_gradient_targets_and_warmup_are_frozen() -> None:
    assert exp71.CALIBRATION_BATCHES == 5
    assert exp71.TARGET_RATE_GRAD_RATIO == 0.025
    assert exp71.TARGET_PERSIST_GRAD_RATIO == 0.05
    assert exp71.PERSIST_LONG_WEIGHT == 0.5
    assert exp71.warmup_scale(1) == 0.1
    assert exp71.warmup_scale(5) == 0.5
    assert exp71.warmup_scale(10) == 1.0
    assert exp71.warmup_scale(50) == 1.0


def test_pairing_seed_is_condition_independent() -> None:
    for role in ("model_init", "train_loader", "calibration_loader"):
        expected = exp71.paired_seed(11, role)
        for condition in exp71.CONDITIONS:
            spec = exp71.RunSpec(condition, 11)
            assert exp71.paired_seed(spec.seed, role) == expected


def test_early_stop_cannot_trigger_before_epoch_50() -> None:
    assert not exp71._early_stop_triggered(49, 1)
    assert exp71._early_stop_triggered(50, 1)
    assert not exp71._early_stop_triggered(64, 35)
    assert exp71._early_stop_triggered(65, 35)


def test_source_file_parses() -> None:
    source = REPO_ROOT / "scripts/experiment_7_1_long_tau_antipersistence.py"
    ast.parse(source.read_text(encoding="utf-8"))


def test_slurm_topology_is_nine_cpu_workers_plus_dependency_finalizer() -> None:
    worker = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_1_cpu_array.bash"
    ).read_text(encoding="utf-8")
    finalizer = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_7_1_cpu.bash"
    ).read_text(encoding="utf-8")
    submit = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_7_1_cpu.bash"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --array=0-8%9" in worker
    for text in (worker, finalizer):
        assert "#SBATCH --cpus-per-task=1" in text
        assert "OMP_NUM_THREADS=1" in text
        assert "MKL_NUM_THREADS=1" in text
        assert "OPENBLAS_NUM_THREADS=1" in text
        assert "NUMEXPR_NUM_THREADS=1" in text
    assert "run-one" in worker
    assert " finalize" in finalizer
    assert 'afterok:${worker_job}' in submit


def test_notebook_is_aggregate_only() -> None:
    path = REPO_ROOT / "notebooks/experiment_7_1_long_tau_antipersistence.ipynb"
    text = path.read_text(encoding="utf-8")
    for token in (
        "summary.csv",
        "dynamics_summary.csv",
        "paired_condition_deltas.csv",
        "calibration_summary.csv",
    ):
        assert token in text
    forbidden = (
        "runs.csv",
        "histories/",
        "evaluations/",
        "rasters/",
        "checkpoints/",
        "run_one(",
        "train_one(",
        "torch.optim",
        "subprocess",
        "sbatch",
    )
    for token in forbidden:
        assert token not in text
