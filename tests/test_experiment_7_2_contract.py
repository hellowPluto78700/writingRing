from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_7_2_two_layer_tau_training as exp72

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_grid_is_84_paired_runs() -> None:
    specs = exp72.run_specs()
    assert len(specs) == 84
    assert {s.architecture for s in specs} == set(exp72.ARCHITECTURES)
    assert {s.training_family for s in specs} == set(exp72.TRAINING_FAMILIES)
    assert {s.regularization for s in specs} == set(exp72.REGULARIZATION_CONDITIONS)
    assert {s.seed for s in specs} == {11, 23, 37}


def test_architecture_matrix_is_frozen() -> None:
    assert exp72.ARCHITECTURES == {
        "234x234": ((2, 3, 4), (2, 3, 4)),
        "34x234": ((3, 4), (2, 3, 4)),
        "34x34": ((3, 4), (3, 4)),
        "34x345": ((3, 4), (3, 4, 5)),
        "34x45": ((3, 4), (4, 5)),
        "4x4": ((4,), (4,)),
        "5x5": ((5,), (5,)),
    }
    assert exp72.HIDDEN_WIDTH == 128
    assert exp72.MAX_EPOCHS == 100
    assert exp72.MIN_EPOCHS == 20
    assert exp72.PATIENCE == 30


def test_shift_group_ranges_are_contiguous() -> None:
    assert exp72.shift_groups((2, 3, 4)) == [
        {"shift": 2, "start": 0, "stop": 43, "count": 43},
        {"shift": 3, "start": 43, "stop": 86, "count": 43},
        {"shift": 4, "start": 86, "stop": 128, "count": 42},
    ]
    assert exp72.shift_groups((3, 4)) == [
        {"shift": 3, "start": 0, "stop": 64, "count": 64},
        {"shift": 4, "start": 64, "stop": 128, "count": 64},
    ]


def test_tau_values_are_expected_at_64_hz() -> None:
    taus = [exp72.tau_ms_from_shift(s, 64.0) for s in (2, 3, 4, 5)]
    assert np.allclose(taus, [54.31342963722199, 117.0136826471659, 242.10347130039656, 492.1461612435418])


def test_training_families_have_distinct_heads_and_probe_points() -> None:
    e2e = exp72.TwoLayerTauSNN(((3, 4), (4, 5)), exp72.FAMILY_E2E, 12, 64.0)
    local = exp72.TwoLayerTauSNN(((3, 4), (4, 5)), exp72.FAMILY_LOCAL, 12, 64.0)
    assert e2e.output_linear is not None and e2e.output_lif is not None and e2e.analog_head is None
    assert local.output_linear is None and local.output_lif is None and local.analog_head is not None
    out = e2e.forward_trajectory(torch.zeros(2, 4, 30))
    assert out["hidden_spikes"][0].shape == (2, 4, 128)
    assert out["hidden_spikes"][1].shape == (2, 4, 128)
    assert out["pre_output_evidence"].shape == (2, 4, 12)
    assert out["output_spikes"].shape == (2, 4, 12)


def test_local_forward_uses_temporary_analog_head() -> None:
    model = exp72.TwoLayerTauSNN(((2, 3, 4), (2, 3, 4)), exp72.FAMILY_LOCAL, 12, 64.0)
    out = model.forward_trajectory(torch.zeros(2, 4, 30))
    assert out["analog_logits"].shape == (2, 4, 12)
    assert "output_spikes" not in out


def test_all_masked_regularizer_ignores_padding_and_averages_layers() -> None:
    spikes = torch.tensor([[[1.0], [0.0], [1.0], [1.0]]])
    assert torch.isclose(exp72.masked_rate_loss(spikes, torch.tensor([2])), torch.tensor(0.5))
    all_ones = torch.ones(1, 3, 1)
    assert torch.isclose(exp72.masked_persistence_loss(all_ones, torch.tensor([2]), 3, 2), torch.tensor(0.0))
    l1, l2 = torch.ones(1, 8, 2), torch.zeros(1, 8, 2)
    rate, _, _, _ = exp72.regularization_terms((l1, l2), torch.tensor([8]))
    assert torch.isclose(rate, torch.tensor(0.5))


def test_regularizer_calibration_contract() -> None:
    assert exp72.CALIBRATION_BATCHES == 5
    assert exp72.TARGET_RATE_GRAD_RATIO == 0.025
    assert exp72.TARGET_PERSIST_GRAD_RATIO == 0.05
    assert exp72.PERSIST_LONG_WEIGHT == 0.5
    assert exp72.warmup_scale(1) == 0.1
    assert exp72.warmup_scale(10) == 1.0


def test_pairing_seed_is_condition_independent() -> None:
    for role in ("model_init", "train_loader", "calibration_loader"):
        expected = exp72.paired_seed(11, role)
        for spec in exp72.run_specs():
            if spec.seed == 11:
                assert exp72.paired_seed(spec.seed, role) == expected


def test_early_stop_cannot_trigger_before_epoch_50() -> None:
    assert not exp72._early_stop_triggered(49, 1)
    assert exp72._early_stop_triggered(50, 1)
    assert not exp72._early_stop_triggered(64, 35)
    assert exp72._early_stop_triggered(65, 35)


def test_source_file_parses() -> None:
    source = REPO_ROOT / "scripts/experiment_7_2_two_layer_tau_training.py"
    ast.parse(source.read_text(encoding="utf-8"))


def test_slurm_topology_is_84_cpu_workers_plus_dependency_finalizer() -> None:
    worker = (REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_2_cpu_array.bash").read_text(encoding="utf-8")
    finalizer = (REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_7_2_cpu.bash").read_text(encoding="utf-8")
    submit = (REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_7_2_cpu.bash").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-83%50" in worker
    for text in (worker, finalizer):
        assert "#SBATCH --cpus-per-task=1" in text
        for token in ("OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"):
            assert token in text
    assert "run-one" in worker
    assert " finalize" in finalizer
    assert 'afterok:${worker_job}' in submit


def test_notebook_is_aggregate_only() -> None:
    text = (REPO_ROOT / "notebooks/experiment_7_2_two_layer_tau_training.ipynb").read_text(encoding="utf-8")
    for token in ("architecture_table.csv", "performance_summary.csv", "dynamics_summary.csv", "paired_deltas.csv", "calibration_summary.csv"):
        assert token in text
    for token in ("performance_runs.csv", "dynamics_runs.csv", "paired_delta_runs.csv", "calibration_runs.csv", "histories/", "evaluations/", "rasters/", "checkpoints/", "torch.optim", "sbatch"):
        assert token not in text
