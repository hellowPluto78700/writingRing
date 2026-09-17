from pathlib import Path

import torch

from scripts import experiment_8_0_3_phase_aware_hierarchical_readout as exp803
from scripts import experiment_8_0_4_phase_aware_hierarchical_readout_mean as exp804


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_matrix_and_backbone_contract() -> None:
    assert exp804.PROTOCOL_VERSION == "phase_aware_hierarchical_readout_mean_v1"
    assert exp804.ARCHITECTURE == "234x234"
    assert exp804.ARCHITECTURE_SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp804.SEEDS == (11, 23, 37)
    assert exp804.METHODS == exp803.METHODS
    assert exp804.EXPECTED_RUNS == 12
    assert len(exp804.run_specs()) == 12


def _synthetic_trajectory(model: exp804.Exp804Net, batch: int = 3, steps: int = 32):
    torch.manual_seed(17)
    l1 = torch.rand(batch, steps, exp804.exp803.exp80.HIDDEN_WIDTH)
    l2 = torch.rand(batch, steps, exp804.exp803.exp80.HIDDEN_WIDTH)
    return {
        "hidden_spikes": [l1, l2],
        "evidence": model.output_linear(l2),
    }


def test_native_objective_is_valid_mean_not_raw_sum() -> None:
    model = exp804.Exp804Net(
        "l2_only_count", n_classes=3, fs=64.0, n_steps=8, bin_steps=4
    )
    trajectory = _synthetic_trajectory(model, batch=2, steps=8)
    lengths = torch.tensor([8, 5], dtype=torch.long)
    evidence = exp803._native_evidence(model, trajectory)
    score = exp804._scores(model, trajectory, lengths)
    expected_mean = exp804.exp803.exp80._valid_mean(evidence, lengths)
    torch.testing.assert_close(score, expected_mean)
    raw_sum = exp803._valid_sum(evidence, lengths)
    assert not torch.allclose(score, raw_sum)


def test_sum_and_mean_have_same_frozen_argmax_but_different_ce_scale() -> None:
    model = exp804.Exp804Net(
        "l1_fixed250_l2_whole_count",
        n_classes=5,
        fs=64.0,
        n_steps=32,
        bin_steps=16,
    )
    trajectory = _synthetic_trajectory(model)
    lengths = torch.tensor([32, 27, 19], dtype=torch.long)
    evidence = exp803._native_evidence(model, trajectory)
    raw_sum = exp803._valid_sum(evidence, lengths)
    valid_mean = exp804._scores(model, trajectory, lengths)
    assert torch.equal(raw_sum.argmax(dim=1), valid_mean.argmax(dim=1))
    y = torch.tensor([0, 1, 2], dtype=torch.long)
    assert not torch.allclose(
        torch.nn.functional.cross_entropy(raw_sum, y),
        torch.nn.functional.cross_entropy(valid_mean, y),
    )


def test_per_timestep_and_explicit_mean_scores_are_equivalent() -> None:
    lengths = torch.tensor([32, 27, 19], dtype=torch.long)
    for method in exp804.METHODS:
        torch.manual_seed(3)
        model = exp804.Exp804Net(
            method, n_classes=5, fs=64.0, n_steps=32, bin_steps=16
        )
        trajectory = _synthetic_trajectory(model)
        accumulated = exp804._scores(model, trajectory, lengths)
        explicit = exp804._explicit_feature_scores(model, trajectory, lengths)
        torch.testing.assert_close(accumulated, explicit, atol=2e-5, rtol=2e-5)


def test_phase_and_no_phase_control_still_match_parameter_count() -> None:
    torch.manual_seed(1)
    phase = exp804.Exp804Net(
        "l1_fixed250_l2_whole_count",
        n_classes=12,
        fs=64.0,
        n_steps=256,
        bin_steps=16,
    )
    torch.manual_seed(1)
    control = exp804.Exp804Net(
        "l1_capacity_no_phase_l2_whole_count",
        n_classes=12,
        fs=64.0,
        n_steps=256,
        bin_steps=16,
    )
    assert sum(p.numel() for p in phase.parameters()) == sum(
        p.numel() for p in control.parameters()
    )


def test_cpu_array_and_dependency_chain() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    run = (root / "run_exp_8_0_4_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-11%12" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "MKL_NUM_THREADS=1" in run
    assert "OPENBLAS_NUM_THREADS=1" in run
    assert "NUMEXPR_NUM_THREADS=1" in run
    assert "experiment_8_0_4_phase_aware_hierarchical_readout_mean" in run

    submit = (root / "submit_exp_8_0_4_cpu.bash").read_text()
    assert 'afterok:${array_job}' in submit
    assert "run_exp_8_0_4_cpu_array.bash" in submit
    assert "finalize_exp_8_0_4_cpu.bash" in submit


def test_plan_and_notebook_contract() -> None:
    plan = (
        REPO_ROOT
        / "docs"
        / "plans"
        / "EXP8_0_4_PHASE_AWARE_MEAN_NORMALIZED_READOUT.md"
    ).read_text()
    assert "4 methods x 3 seeds = 12 independent CPU jobs" in plan
    assert "valid-length-mean CE" in plan
    assert "l1_fixed250_l2_whole_count" in plan
    assert "single" in plan.lower()

    notebook = (
        REPO_ROOT
        / "notebooks"
        / "experiment_8_0_4_phase_aware_hierarchical_readout_mean.ipynb"
    ).read_text()
    assert "phase_aware_hierarchical_readout_mean_v1" in notebook
    assert "method_summary.csv" in notebook
    assert "probe_summary.csv" in notebook
    assert "paired_delta_summary.csv" in notebook
    assert "phase_structure_summary.csv" in notebook
    assert "history_runs.csv" in notebook
