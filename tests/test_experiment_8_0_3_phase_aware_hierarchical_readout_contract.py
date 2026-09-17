from pathlib import Path

import torch

from scripts import experiment_8_0_3_phase_aware_hierarchical_readout as exp803


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_matrix_and_backbone_contract() -> None:
    assert exp803.PROTOCOL_VERSION == "phase_aware_hierarchical_readout_v2"
    assert exp803.ARCHITECTURE == "234x234"
    assert exp803.ARCHITECTURE_SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp803.SEEDS == (11, 23, 37)
    assert exp803.METHODS == (
        "l2_only_count",
        "l1_l2_timeshared_count",
        "l1_fixed250_l2_whole_count",
        "l1_capacity_no_phase_l2_whole_count",
    )
    assert exp803.EXPECTED_RUNS == 12
    assert len(exp803.run_specs()) == 12


def test_phase_and_capacity_control_have_equal_parameter_count() -> None:
    torch.manual_seed(1)
    phase = exp803.Exp803Net(
        "l1_fixed250_l2_whole_count",
        n_classes=12,
        fs=64.0,
        n_steps=256,
        bin_steps=16,
    )
    torch.manual_seed(1)
    control = exp803.Exp803Net(
        "l1_capacity_no_phase_l2_whole_count",
        n_classes=12,
        fs=64.0,
        n_steps=256,
        bin_steps=16,
    )
    assert phase.n_bins == 16
    assert phase.l1_phase_linear is not None
    assert control.l1_phase_linear is not None
    assert phase.l1_phase_linear.weight.shape == (12, 16 * exp803.exp80.HIDDEN_WIDTH)
    assert phase.l1_phase_linear.weight.shape == control.l1_phase_linear.weight.shape
    assert sum(p.numel() for p in phase.parameters()) == sum(
        p.numel() for p in control.parameters()
    )


def _synthetic_trajectory(model: exp803.Exp803Net, batch: int = 3, steps: int = 32):
    torch.manual_seed(7)
    l1 = torch.rand(batch, steps, exp803.exp80.HIDDEN_WIDTH)
    l2 = torch.rand(batch, steps, exp803.exp80.HIDDEN_WIDTH)
    return {
        "hidden_spikes": [l1, l2],
        "evidence": model.output_linear(l2),
    }


def test_per_timestep_and_explicit_count_scores_are_equivalent() -> None:
    lengths = torch.tensor([32, 27, 19], dtype=torch.long)
    for method in exp803.METHODS:
        torch.manual_seed(3)
        model = exp803.Exp803Net(
            method, n_classes=5, fs=64.0, n_steps=32, bin_steps=16
        )
        trajectory = _synthetic_trajectory(model)
        accumulated = exp803._scores(model, trajectory, lengths)
        explicit = exp803._explicit_feature_scores(model, trajectory, lengths)
        torch.testing.assert_close(accumulated, explicit, atol=2e-5, rtol=2e-5)


def test_native_objective_is_sum_not_valid_mean() -> None:
    model = exp803.Exp803Net(
        "l2_only_count", n_classes=3, fs=64.0, n_steps=8, bin_steps=4
    )
    trajectory = _synthetic_trajectory(model, batch=2, steps=8)
    lengths = torch.tensor([8, 5], dtype=torch.long)
    evidence = exp803._native_evidence(model, trajectory)
    score = exp803._scores(model, trajectory, lengths)
    expected_sum = exp803._valid_sum(evidence, lengths)
    torch.testing.assert_close(score, expected_sum)
    mean = exp803.exp80._valid_mean(evidence, lengths)
    assert not torch.allclose(score, mean)


def test_phase_head_changes_weight_with_absolute_bin() -> None:
    model = exp803.Exp803Net(
        "l1_fixed250_l2_whole_count",
        n_classes=2,
        fs=64.0,
        n_steps=32,
        bin_steps=16,
    )
    assert model.l1_phase_linear is not None
    with torch.no_grad():
        model.l1_phase_linear.weight.zero_()
        bank = model.phase_weight_bank()
        bank[0, 0, 0] = 1.0
        bank[1, 0, 0] = 3.0
    l1 = torch.zeros(1, 32, exp803.exp80.HIDDEN_WIDTH)
    l1[0, 0, 0] = 1.0
    l1[0, 16, 0] = 1.0
    evidence = exp803._l1_branch_evidence(model, l1)
    assert evidence[0, 0, 0].item() == 1.0
    assert evidence[0, 16, 0].item() == 3.0


def test_capacity_control_has_no_phase_access_and_uses_sqrt_scaling() -> None:
    model = exp803.Exp803Net(
        "l1_capacity_no_phase_l2_whole_count",
        n_classes=2,
        fs=64.0,
        n_steps=32,
        bin_steps=16,
    )
    assert model.l1_phase_linear is not None
    with torch.no_grad():
        model.l1_phase_linear.weight.zero_()
        bank = model.phase_weight_bank()
        bank[:, 0, 0] = 1.0
    l1 = torch.zeros(1, 32, exp803.exp80.HIDDEN_WIDTH)
    l1[0, 0, 0] = 1.0
    l1[0, 16, 0] = 1.0
    evidence = exp803._l1_branch_evidence(model, l1)
    expected = (model.n_bins ** 0.5)
    assert evidence[0, 0, 0].item() == expected
    assert evidence[0, 16, 0].item() == expected


def test_cpu_array_and_dependency_chain() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    run = (root / "run_exp_8_0_3_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-11%12" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "MKL_NUM_THREADS=1" in run
    assert "OPENBLAS_NUM_THREADS=1" in run
    assert "NUMEXPR_NUM_THREADS=1" in run
    assert "--array-task-id" in run
    assert "experiment_8_0_3_phase_aware_hierarchical_readout" in run

    submit = (root / "submit_exp_8_0_3_cpu.bash").read_text()
    assert 'afterok:${array_job}' in submit
    assert "run_exp_8_0_3_cpu_array.bash" in submit
    assert "finalize_exp_8_0_3_cpu.bash" in submit


def test_plan_and_notebook_contract() -> None:
    plan = (
        REPO_ROOT / "docs" / "plans" / "EXP8_0_3_PHASE_AWARE_HIERARCHICAL_READOUT.md"
    ).read_text()
    assert "4 methods x 3 seeds = 12 independent CPU jobs" in plan
    assert "l1_fixed250_l2_whole_count" in plan
    assert "l1_capacity_no_phase_l2_whole_count" in plan
    assert "count-form" in plan
    assert "phase access" in plan

    notebook = (
        REPO_ROOT / "notebooks" / "experiment_8_0_3_phase_aware_hierarchical_readout.ipynb"
    ).read_text()
    assert "phase_aware_hierarchical_readout_v2" in notebook
    assert "method_summary.csv" in notebook
    assert "probe_summary.csv" in notebook
    assert "paired_delta_summary.csv" in notebook
    assert "phase_structure_summary.csv" in notebook
    assert "history_runs.csv" in notebook
