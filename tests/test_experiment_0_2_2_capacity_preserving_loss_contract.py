from pathlib import Path

import numpy as np
import torch

from scripts import experiment_0_2_2_capacity_preserving_loss as exp022


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_matrices_are_exact_and_unique() -> None:
    warmups = exp022.warmup_specs()
    runs = exp022.run_specs()
    assert len(warmups) == 6
    assert len({spec.key for spec in warmups}) == 6
    assert len(runs) == 30
    assert len({spec.key for spec in runs}) == 30
    assert {spec.architecture for spec in runs} == {"mid_long", "short_mid_long"}
    assert {spec.condition for spec in runs} == {
        "wc_only",
        "sat",
        "relative_tail",
        "sat_relative_tail",
        "sat_relative_tail_capacity",
    }
    assert {spec.seed for spec in runs} == {11, 23, 37}
    assert exp022.OBJECTIVE == "whole_count_ce"
    assert exp022.EPOCHS == 50


def test_all_conditions_share_one_warmup_per_backbone_seed() -> None:
    for architecture in exp022.ARCHITECTURES:
        for seed in exp022.SEEDS:
            matching = [spec for spec in exp022.run_specs() if spec.architecture == architecture and spec.seed == seed]
            assert len(matching) == len(exp022.CONDITIONS)
            warmup_keys = {spec.warmup_spec.key for spec in matching}
            assert warmup_keys == {exp022.WarmupSpec(architecture, seed).key}


def test_regularizer_schedule_is_zero_then_ramps_then_stays_full() -> None:
    assert exp022.regularizer_scale(1) == 0.0
    assert exp022.regularizer_scale(5) == 0.0
    assert np.isclose(exp022.regularizer_scale(6), 0.1)
    assert np.isclose(exp022.regularizer_scale(10), 0.5)
    assert exp022.regularizer_scale(15) == 1.0
    assert exp022.regularizer_scale(50) == 1.0


def test_long_regularizers_target_only_s6_s7() -> None:
    assert exp022.LONG_SHIFTS == (6, 7)
    for architecture in exp022.ARCHITECTURES:
        selected = exp022._selected_slice_indices(architecture, exp022.LONG_SHIFTS)
        assert [shift for shift, _, _ in selected] == [6, 7]
        assert all(stop > start for _, start, stop in selected)
        mapping = exp022._shift_slices(architecture)
        assert set(mapping) >= {2, 3, 4, 5, 6, 7}


def test_saturation_loss_penalizes_persistent_not_sparse_firing() -> None:
    length = torch.tensor([20], dtype=torch.long)
    persistent = torch.zeros(1, 20, 128)
    sparse = torch.zeros_like(persistent)
    for _, start, stop in exp022._selected_slice_indices("mid_long", exp022.LONG_SHIFTS):
        persistent[:, :, start:stop] = 1.0
        sparse[:, 0, start:stop] = 1.0
    persistent_loss = exp022.saturation_loss(persistent, length, "mid_long", 64.0, rho_max=0.5)
    sparse_loss = exp022.saturation_loss(sparse, length, "mid_long", 64.0, rho_max=0.5)
    assert persistent_loss.item() > 0.0
    assert sparse_loss.item() == 0.0


def test_relative_tail_loss_rejects_persistent_tail_but_not_silent_tail() -> None:
    lengths = torch.tensor([20], dtype=torch.long)
    n_steps = 20 + exp022.exp02.tail_horizon_steps(64.0)
    persistent = torch.zeros(1, n_steps, 128)
    silent_tail = torch.zeros_like(persistent)
    for _, start, stop in exp022._selected_slice_indices("short_mid_long", exp022.LONG_SHIFTS):
        persistent[:, :, start:stop] = 1.0
        silent_tail[:, :20, start:stop] = 1.0
    gamma = (0.5, 0.5, 0.5)
    bad = exp022.relative_tail_loss(persistent, lengths, "short_mid_long", 64.0, gamma)
    good = exp022.relative_tail_loss(silent_tail, lengths, "short_mid_long", 64.0, gamma)
    assert bad.item() > 0.0
    assert good.item() == 0.0


def test_capacity_floor_prevents_global_long_population_shutdown() -> None:
    lengths = torch.tensor([20], dtype=torch.long)
    indices = torch.tensor([0], dtype=torch.long)
    silent = torch.zeros(1, 20, 128)
    active = torch.zeros_like(silent)
    for _, start, stop in exp022._selected_slice_indices("mid_long", exp022.LONG_SHIFTS):
        active[:, :, start:stop] = 1.0
    warmup = {6: torch.tensor([10.0]), 7: torch.tensor([10.0])}
    silent_loss = exp022.capacity_floor_loss(silent, lengths, indices, "mid_long", 64.0, warmup)
    active_loss = exp022.capacity_floor_loss(active, lengths, indices, "mid_long", 64.0, warmup)
    assert silent_loss.item() > 0.0
    assert active_loss.item() == 0.0
    assert exp022.CAPACITY_ETA == 0.70


def test_history_schema_contains_requested_ba_loss_reg_and_dynamics() -> None:
    required = {
        "train_balanced_accuracy",
        "val_balanced_accuracy",
        "train_task_ce",
        "val_task_ce",
        "train_optim_task_loss",
        "train_optim_raw_reg_loss",
        "train_optim_weighted_reg_loss",
        "train_optim_total_loss",
        "train_sat_loss",
        "train_relative_tail_loss",
        "train_capacity_loss",
        "train_long_valid_fr_hz",
        "train_long_tail_fr_hz",
        "val_long_valid_fr_hz",
        "val_long_tail_fr_hz",
        "output_weight_norm_s6",
        "output_weight_norm_s7",
        "output_weight_energy_long_fraction",
    }
    assert required.issubset(set(exp022.HISTORY_COLUMNS))


def test_healthy_reference_is_train_only_and_uses_short_mid_s4_s5() -> None:
    assert exp022.HEALTHY_REFERENCE_ARCHITECTURE == "short_mid"
    assert exp022.HEALTHY_REFERENCE_SHIFTS == (4, 5)
    assert exp022.SATURATION_QUANTILE == 0.95
    assert exp022.RELATIVE_TAIL_QUANTILE == 0.90


def test_slurm_topology_uses_task_level_one_core_parallelism() -> None:
    warmup = (REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_0_2_2_warmup_cpu_array.bash").read_text(encoding="utf-8")
    main = (REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_0_2_2_capacity_preserving_cpu_array.bash").read_text(encoding="utf-8")
    submit = (REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_0_2_2_capacity_preserving_cpu.bash").read_text(encoding="utf-8")
    finalize = (REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_0_2_2_capacity_preserving_cpu.bash").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-5%6" in warmup
    assert "#SBATCH --array=0-29%30" in main
    assert "#SBATCH --cpus-per-task=1" in warmup
    assert "#SBATCH --cpus-per-task=1" in main
    for text in (warmup, main):
        assert "OMP_NUM_THREADS=1" in text
        assert "MKL_NUM_THREADS=1" in text
        assert "OPENBLAS_NUM_THREADS=1" in text
        assert "NUMEXPR_NUM_THREADS=1" in text
    assert "afterok:${ref_job}:${warmup_job}" in submit
    assert "afterok:${main_job}" in submit
    assert " finalize" in finalize
    assert "run-one" not in finalize


def test_notebook_is_analysis_only_and_reads_finalized_histories() -> None:
    text = (REPO_ROOT / "notebooks/experiment_0_2_2_capacity_preserving_loss.ipynb").read_text(encoding="utf-8")
    forbidden = ["torch.load", "DataLoader", "optimizer", "backward(", "multiprocessing", "subprocess"]
    assert not any(token in text for token in forbidden)
    assert "history_long.csv" in text
    assert "comparison_summary.csv" in text
    assert "train_optim_weighted_reg_loss" in text
    assert "train_optim_total_loss" in text
