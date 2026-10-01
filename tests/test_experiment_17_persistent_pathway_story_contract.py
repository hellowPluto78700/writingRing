from __future__ import annotations

from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import torch

from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_16_3_run_reward as exp16_3
from scripts import experiment_17_persistent_pathway_story as exp


def _p():
    return replace(
        smoke_protocol(),
        width=12,
        input_channels=3,
        total_channels=3,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=4,
        max_epochs=7,
        min_epochs=1,
        patience=2,
        c_grid=(0.1, 1.0),
        probe_max_iter=200,
    )


def test_manifest_axes_are_locked():
    assert exp.FORMAL_SEEDS == (11, 23, 37)
    assert exp.CHECKPOINT_EVERY == 5
    assert exp.PERSISTENT_QUARTILE == 0.25
    assert exp.LONGITUDINAL_PRUNE_FRACTION == 0.30
    assert exp.SUBSET_FRACTION == 0.30
    assert exp.PRUNE_FRACTIONS == (0.10, 0.20, 0.30)
    assert exp.SUBSET_RANDOM_REPLICATES == 5
    assert exp.PRUNE_RANDOM_REPLICATES == 20
    assert len(exp.trajectory_specs()) == 3
    assert len(exp.artifact_specs()) == 24
    for seed in exp.FORMAL_SEEDS:
        assert [s.case for s in exp.artifact_specs() if s.seed == seed] == list(exp.SOURCE_CASES)


def test_trajectory_initialization_is_exact_exp16_3_lin():
    p = _p()
    for seed in exp.FORMAL_SEEDS:
        a = exp._make_trajectory_model(seed, p).state_dict()
        b = exp16_3._make_model(exp16_3.ExpSpec("LIN", seed, "linear", None), p).state_dict()
        assert a.keys() == b.keys()
        for name in a:
            assert torch.equal(a[name], b[name]), (seed, name)


def test_trajectory_training_selection_never_reads_test():
    source = inspect.getsource(exp._train_trajectory)
    assert '"val"' in source
    assert '"test"' not in source
    assert "test_ba" not in source
    assert "% CHECKPOINT_EVERY" in source
    assert "selected_best" in source
    assert "stopped" in source


def test_dimension_matched_subset_masks_are_deterministic():
    occupancy = np.linspace(0.0, 1.0, 12)
    first = exp._subset_specs(occupancy, 11, "C0")
    second = exp._subset_specs(occupancy, 11, "C0")
    assert len(first) == 2 + exp.SUBSET_RANDOM_REPLICATES
    expected_k = int(np.ceil(exp.SUBSET_FRACTION * len(occupancy)))
    for (kind_a, rep_a, idx_a), (kind_b, rep_b, idx_b) in zip(first, second):
        assert (kind_a, rep_a) == (kind_b, rep_b)
        assert len(idx_a) == expected_k
        assert len(np.unique(idx_a)) == expected_k
        assert np.array_equal(idx_a, idx_b)
    high = first[0][2]
    low = first[1][2]
    assert set(high).isdisjoint(set(low))
    assert np.array_equal(high, np.array([8, 9, 10, 11]))
    assert np.array_equal(low, np.array([0, 1, 2, 3]))


def test_user_cv_has_each_user_in_every_fold_and_is_deterministic():
    users = np.repeat(np.array(["u0", "u1", "u2"]), 10)
    a = exp._user_cv_fold_ids(users, 23)
    b = exp._user_cv_fold_ids(users, 23)
    assert np.array_equal(a, b)
    for user in np.unique(users):
        counts = np.bincount(a[users == user], minlength=exp.USER_CV_FOLDS)
        assert np.array_equal(counts, np.array([2, 2, 2, 2, 2]))


def test_class_residualization_uses_training_centroids():
    x_train = np.array([[1.0, 0.0], [3.0, 2.0], [10.0, 4.0], [14.0, 8.0]])
    y_train = np.array([0, 0, 1, 1])
    x_val = np.array([[5.0, 2.0], [12.0, 6.0]])
    y_val = np.array([0, 1])
    x_test = np.array([[2.0, 1.0], [16.0, 9.0]])
    y_test = np.array([0, 1])
    train, val, test = exp._class_residualized_triplet(
        x_train, y_train, x_val, y_val, x_test, y_test
    )
    assert np.allclose(train[y_train == 0].mean(0), 0.0)
    assert np.allclose(train[y_train == 1].mean(0), 0.0)
    # Class-0 train centroid is [2,1], class-1 is [12,6].
    assert np.allclose(val, np.array([[3.0, 1.0], [0.0, 0.0]]))
    assert np.allclose(test, np.array([[0.0, 0.0], [4.0, 3.0]]))


def test_pruning_rankings_are_training_only_inputs_and_weight_ordered():
    occupancy = np.array([0.2, 0.9, 0.1, 0.5])
    probe = SimpleNamespace(
        coef_=np.array([[1.0, 0.0, 3.0, 0.5], [0.0, 2.0, 0.0, 0.5]])
    )
    rankings = exp._pruning_rankings(occupancy, probe)
    assert np.array_equal(rankings["high_occupancy"], np.array([1, 3, 0, 2]))
    assert np.array_equal(rankings["low_occupancy"], np.array([2, 0, 3, 1]))
    assert np.array_equal(rankings["high_readout_weight"], np.array([2, 1, 0, 3]))


def test_random_pruning_mask_is_dimension_matched_and_reproducible():
    a = exp._random_subset_indices(128, 39, 37, "S70", "prune-0.3", 9)
    b = exp._random_subset_indices(128, 39, 37, "S70", "prune-0.3", 9)
    c = exp._random_subset_indices(128, 39, 37, "S70", "prune-0.3", 10)
    assert len(a) == 39
    assert len(np.unique(a)) == 39
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_slurm_contract():
    root = exp.find_repo_root()
    bash = root / "scripts" / "bash_script" / "SNN_Bash"
    trajectory = (bash / "run_exp_17_trajectory_cpu_array.bash").read_text()
    subset = (bash / "run_exp_17_subset_cpu_array.bash").read_text()
    pruning = (bash / "run_exp_17_pruning_cpu_array.bash").read_text()
    submit = (bash / "submit_exp_17_cpu.bash").read_text()
    assert "#SBATCH --array=0-2%3" in trajectory
    assert "#SBATCH --array=0-23%24" in subset
    assert "#SBATCH --array=0-23%24" in pruning
    for source in (trajectory, subset, pruning):
        assert "#SBATCH --cpus-per-task=1" in source
        assert "source /etc/profile" in source
        assert "OMP_NUM_THREADS=1" in source
    assert "--dependency=afterok:" in submit
    assert "jid_traj" in submit
    assert "jid_subset" in submit
    assert "jid_prune" in submit