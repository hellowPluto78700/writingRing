from pathlib import Path

import torch

from scripts import experiment_8_0_local_backbone_tau_sweep as exp80


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_run_matrix() -> None:
    assert exp80.SEEDS == (11, 23, 37)
    assert exp80.ARCHITECTURE_ORDER == (
        "234x234",
        "123x234",
        "123x123",
        "123x345",
        "234x345",
    )
    assert exp80.ARCHITECTURES["234x234"] == ((2, 3, 4), (2, 3, 4))
    assert exp80.ARCHITECTURES["123x234"] == ((1, 2, 3), (2, 3, 4))
    assert exp80.ARCHITECTURES["123x123"] == ((1, 2, 3), (1, 2, 3))
    assert exp80.ARCHITECTURES["123x345"] == ((1, 2, 3), (3, 4, 5))
    assert exp80.ARCHITECTURES["234x345"] == ((2, 3, 4), (3, 4, 5))
    assert exp80.EXPECTED_RUNS == 15
    assert len(exp80.run_specs()) == 15


def test_a2_style_hidden_and_linear_head_contract() -> None:
    model = exp80.Exp80Net(((1, 2, 3), (2, 3, 4)), n_classes=12, fs=64.0)
    assert len(model.hidden_linears) == 2
    assert model.hidden_linears[0].in_features == 30
    assert model.hidden_linears[0].out_features == 128
    assert model.hidden_linears[0].bias is None
    assert model.hidden_linears[1].in_features == 128
    assert model.hidden_linears[1].out_features == 128
    assert model.hidden_linears[1].bias is None
    assert model.output_linear.in_features == 128
    assert model.output_linear.out_features == 12
    assert model.output_linear.bias is None

    x = torch.zeros(2, 8, 30)
    tr = model.forward_trajectory(x)
    assert tr["hidden_spikes"][0].shape == (2, 8, 128)
    assert tr["hidden_spikes"][1].shape == (2, 8, 128)
    assert tr["evidence"].shape == (2, 8, 12)


def test_output_lif_transfer_contract() -> None:
    assert exp80.OUTPUT_ALPHA == 0.0
    assert exp80.OUTPUT_BETA == 0.5
    assert exp80.OUTPUT_CAP == 1
    evidence = torch.zeros(3, 10, 12)
    spikes = exp80._output_lif_spikes(evidence)
    assert spikes.shape == evidence.shape
    assert torch.count_nonzero(spikes).item() == 0


def test_shift_groups_are_contiguous_and_cover_layer() -> None:
    groups = exp80.shift_groups((1, 2, 3))
    assert [(g["start"], g["stop"]) for g in groups] == [(0, 43), (43, 86), (86, 128)]
    assert [g["count"] for g in groups] == [43, 43, 42]


def test_slurm_parallelization_and_dependency_chain() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    run = (root / "run_exp_8_0_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-14%15" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "MKL_NUM_THREADS=1" in run
    assert "OPENBLAS_NUM_THREADS=1" in run
    assert "--array-task-id" in run

    submit = (root / "submit_exp_8_0_cpu.bash").read_text()
    assert 'afterok:${array_job}' in submit
    assert "run_exp_8_0_cpu_array.bash" in submit
    assert "finalize_exp_8_0_cpu.bash" in submit


def test_plan_and_notebook_contract() -> None:
    plan = (REPO_ROOT / "docs" / "plans" / "EXP8_0_LOCAL_BACKBONE_TAU_SWEEP.md").read_text()
    assert "5 architectures x 3 seeds = 15 training jobs" in plan
    assert "L1 neurons x timestep" in plan
    assert "L2 neurons x timestep" in plan
    assert "12 output-LIF neurons x timestep" in plan

    notebook = (REPO_ROOT / "notebooks" / "experiment_8_0_local_backbone_tau_sweep.ipynb").read_text()
    assert "method_summary.csv" in notebook
    assert "contrast_summary.csv" in notebook
    assert "activity_summary.csv" in notebook
    assert "raster_index.csv" in notebook
    assert "Per-run raster appendix" in notebook
