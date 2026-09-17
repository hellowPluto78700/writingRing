from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from scripts import experiment_8_0_5_frozen_backbone_phase_readout as exp805


REPO_ROOT = Path(__file__).resolve().parents[1]


def _base() -> exp805.BaseFeatures:
    l1_fixed = torch.zeros(4, 4, 3)
    for sample in range(4):
        for bin_index in range(4):
            l1_fixed[sample, bin_index] = (
                torch.arange(3, dtype=torch.float32)
                + 10 * bin_index
                + sample
            )
    return exp805.BaseFeatures(
        l1_count=l1_fixed.sum(dim=1),
        l2_count=torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1,
        l1_fixed=l1_fixed,
        lengths=torch.tensor([16, 15, 14, 13]),
        y=torch.tensor([0, 1, 2, 0]),
    )


def test_matrix_and_frozen_source_contract() -> None:
    assert exp805.METHODS == (
        "l2_whole",
        "l1_l2_timeshared",
        "l1_fixed250_l2_whole_true_phase",
        "l1_fixed250_l2_whole_destroyed_phase",
    )
    assert exp805.SEEDS == (11, 23, 37)
    assert exp805.EXPECTED_RUNS == 12
    assert exp805.BACKBONE_METHOD == "l1_l2_timeshared_count"
    assert exp805.ARCHITECTURE == "234x234"


def test_true_and_destroyed_phase_have_same_effective_capacity() -> None:
    base = _base()
    true = exp805.build_head_features(
        base, exp805.TRUE_PHASE_METHOD, 11, "train"
    )
    destroyed = exp805.build_head_features(
        base, exp805.DESTROYED_PHASE_METHOD, 11, "train"
    )
    assert true.feature_dim == destroyed.feature_dim
    assert true.l1_dim == destroyed.l1_dim == base.n_bins * base.hidden_width
    assert true.l2_dim == destroyed.l2_dim == base.hidden_width
    assert exp805.head_init_seed(11, exp805.TRUE_PHASE_METHOD) == exp805.head_init_seed(
        11, exp805.DESTROYED_PHASE_METHOD
    )
    assert not torch.equal(true.x, destroyed.x)


def test_destroyed_phase_offsets_are_deterministic_nonzero() -> None:
    a = exp805.phase_destroy_offsets(20, 16, 23, "train")
    b = exp805.phase_destroy_offsets(20, 16, 23, "train")
    c = exp805.phase_destroy_offsets(20, 16, 23, "test")
    torch.testing.assert_close(a, b)
    assert bool(torch.all(a > 0))
    assert bool(torch.all(a < 16))
    assert not torch.equal(a, c)


def test_global_zero_shift_is_exact_identity() -> None:
    base = _base()
    normal = exp805.build_head_features(
        base, exp805.TRUE_PHASE_METHOD, 37, "test"
    )
    zero = exp805.build_head_features(
        base,
        exp805.TRUE_PHASE_METHOD,
        37,
        "test",
        global_phase_shift=0,
    )
    shifted = exp805.build_head_features(
        base,
        exp805.TRUE_PHASE_METHOD,
        37,
        "test",
        global_phase_shift=1,
    )
    torch.testing.assert_close(normal.x, zero.x)
    assert not torch.equal(normal.x, shifted.x)


def test_branch_decomposition_is_exact() -> None:
    base = _base()
    features = exp805.build_head_features(
        base, exp805.TRUE_PHASE_METHOD, 11, "test"
    )
    head = exp805.new_head(features.feature_dim, 3, 11, exp805.TRUE_PHASE_METHOD)
    full = F.linear(features.x, head.weight)
    l1 = F.linear(
        features.x[:, : features.l1_dim],
        head.weight[:, : features.l1_dim],
    )
    l2 = F.linear(
        features.x[:, features.l1_dim :],
        head.weight[:, features.l1_dim :],
    )
    torch.testing.assert_close(full, l1 + l2)


def test_slurm_and_artifact_contract() -> None:
    run_script = (
        REPO_ROOT
        / "scripts"
        / "bash_script"
        / "SNN_Bash"
        / "run_exp_8_0_5_cpu_array.bash"
    ).read_text()
    submit_script = (
        REPO_ROOT
        / "scripts"
        / "bash_script"
        / "SNN_Bash"
        / "submit_exp_8_0_5_cpu.bash"
    ).read_text()
    plan = (
        REPO_ROOT
        / "docs"
        / "plans"
        / "EXP8_0_5_FROZEN_BACKBONE_PHASE_READOUT.md"
    ).read_text()
    notebook = (
        REPO_ROOT
        / "notebooks"
        / "experiment_8_0_5_frozen_backbone_phase_readout.ipynb"
    ).read_text()

    assert "#SBATCH --array=0-11%12" in run_script
    assert "#SBATCH --cpus-per-task=1" in run_script
    assert "OMP_NUM_THREADS=1" in run_script
    assert 'afterok:${array_job}' in submit_script
    assert "L1 is frozen" in plan
    assert "L2 is frozen" in plan
    assert "true_phase_vs_destroyed_phase" in notebook
    assert "branch_ablation_summary.csv" in notebook
    assert "phase_shift_summary.csv" in notebook
