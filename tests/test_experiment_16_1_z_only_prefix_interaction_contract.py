from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import pandas as pd
import torch

from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_16_1_z_only_prefix_interaction as exp


def _small_protocol():
    return replace(
        smoke_protocol(),
        width=6,
        input_channels=3,
        total_channels=3,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=4,
        max_epochs=3,
        min_epochs=1,
        patience=1,
        relative_bins=4,
        shuffle_seeds=(101,),
    )


def test_exp16_1_manifest_and_task_counts() -> None:
    assert exp.SEEDS == (11, 23, 37)
    assert exp.PREFIX_LAMBDAS == (0.01, 0.03, 0.05)
    assert 0.0 not in exp.PREFIX_LAMBDAS
    assert len(exp.specs()) == 36
    assert len(exp.gated_specs()) == 24
    assert sum(spec.gate_mode == "none" for spec in exp.specs()) == 12
    assert sum(spec.gate_mode == "z_only" for spec in exp.specs()) == 12
    assert sum(spec.gate_mode == "z_history" for spec in exp.specs()) == 12


def test_each_seed_has_complete_factorial() -> None:
    for seed in exp.SEEDS:
        rows = [spec for spec in exp.specs() if spec.seed == seed]
        assert len(rows) == 12
        assert [spec.case for spec in rows[:4]] == [
            "C0",
            "P_lp0p01",
            "P_lp0p03",
            "P_lp0p05",
        ]
        assert [spec.case for spec in rows[4:8]] == [
            "GZ0",
            "GPZ_lp0p01",
            "GPZ_lp0p03",
            "GPZ_lp0p05",
        ]
        assert [spec.case for spec in rows[8:12]] == [
            "GZH0",
            "GPZH_lp0p01",
            "GPZH_lp0p03",
            "GPZH_lp0p05",
        ]


def test_common_backbone_initialization_is_paired() -> None:
    p = _small_protocol()
    c0 = BenchmarkNet(exp._run(exp.ExpSpec("C0", 11, "none", 0.0)), p)
    gz = exp.ContextGateNet(
        exp._run(exp.ExpSpec("GZ0", 11, "z_only", 0.0)),
        p,
        "z_only",
    )
    gzh = exp.ContextGateNet(
        exp._run(exp.ExpSpec("GZH0", 11, "z_history", 0.0)),
        p,
        "z_history",
    )
    for key, value in c0.state_dict().items():
        if key in gz.state_dict():
            assert torch.equal(value, gz.state_dict()[key]), key
        if key in gzh.state_dict():
            assert torch.equal(value, gzh.state_dict()[key]), key


def test_z_only_removes_explicit_history_parameter() -> None:
    p = _small_protocol()
    z = exp.ContextGateNet(
        exp._run(exp.ExpSpec("GZ0", 11, "z_only", 0.0)),
        p,
        "z_only",
    )
    zh = exp.ContextGateNet(
        exp._run(exp.ExpSpec("GZH0", 11, "z_history", 0.0)),
        p,
        "z_history",
    )
    z_names = dict(z.named_parameters())
    zh_names = dict(zh.named_parameters())
    assert "gate_input.weight" in z_names
    assert "gate_history.weight" not in z_names
    assert "gate_history.weight" in zh_names
    assert z.gate_history is None
    assert zh.gate_history is not None


def test_gate_is_true_suppressive_and_initially_point_nine() -> None:
    p = _small_protocol()
    model = exp.ContextGateNet(
        exp._run(exp.ExpSpec("GZ0", 11, "z_only", 0.0)),
        p,
        "z_only",
    )
    x = torch.rand(4, p.steps, p.input_channels)
    lengths = torch.tensor([16, 14, 12, 10])
    with torch.no_grad():
        out = model(x, lengths)
    valid = torch.arange(p.steps)[None, :] < lengths[:, None]
    gate = out["gate_g"][valid]
    assert torch.all(gate > 0)
    assert torch.all(gate < 1)
    assert torch.allclose(gate, torch.full_like(gate, 0.9), atol=1e-6)


def test_parameter_groups_reflect_gate_mode() -> None:
    p = _small_protocol()
    z = exp.ContextGateNet(
        exp._run(exp.ExpSpec("GZ0", 11, "z_only", 0.0)),
        p,
        "z_only",
    )
    zh = exp.ContextGateNet(
        exp._run(exp.ExpSpec("GZH0", 11, "z_history", 0.0)),
        p,
        "z_history",
    )
    assert set(exp._parameter_groups(z)) == {
        "R",
        "W_L1",
        "W_L2",
        "G_z",
        "G_b",
    }
    assert set(exp._parameter_groups(zh)) == {
        "R",
        "W_L1",
        "W_L2",
        "G_z",
        "G_h",
        "G_b",
    }


def test_interaction_formula_is_difference_in_differences() -> None:
    native = pd.DataFrame(
        [
            {"case": "C0", "seed": 11, "val_ba": 0.50, "test_ba": 0.51},
            {"case": "P_lp0p01", "seed": 11, "val_ba": 0.51, "test_ba": 0.52},
            {"case": "GZ0", "seed": 11, "val_ba": 0.52, "test_ba": 0.53},
            {"case": "GPZ_lp0p01", "seed": 11, "val_ba": 0.55, "test_ba": 0.57},
            {"case": "GZH0", "seed": 11, "val_ba": 0.52, "test_ba": 0.53},
            {"case": "GPZH_lp0p01", "seed": 11, "val_ba": 0.54, "test_ba": 0.55},
        ]
    )
    collapse = pd.DataFrame(
        [
            {"case": "C0", "seed": 11, "whole_count": 0.50, "relative10_ordered": 0.60, "collapse_gap_pp": 10.0},
            {"case": "P_lp0p01", "seed": 11, "whole_count": 0.51, "relative10_ordered": 0.61, "collapse_gap_pp": 10.0},
            {"case": "GZ0", "seed": 11, "whole_count": 0.52, "relative10_ordered": 0.62, "collapse_gap_pp": 10.0},
            {"case": "GPZ_lp0p01", "seed": 11, "whole_count": 0.55, "relative10_ordered": 0.63, "collapse_gap_pp": 8.0},
            {"case": "GZH0", "seed": 11, "whole_count": 0.52, "relative10_ordered": 0.62, "collapse_gap_pp": 10.0},
            {"case": "GPZH_lp0p01", "seed": 11, "whole_count": 0.54, "relative10_ordered": 0.63, "collapse_gap_pp": 9.0},
        ]
    )
    old_seeds, old_lambdas = exp.SEEDS, exp.PREFIX_LAMBDAS
    try:
        exp.SEEDS = (11,)
        exp.PREFIX_LAMBDAS = (0.01,)
        rows = exp._interaction_rows(native, collapse).set_index("gate_mode")
    finally:
        exp.SEEDS = old_seeds
        exp.PREFIX_LAMBDAS = old_lambdas
    assert abs(rows.loc["z_only", "val_ba_interaction_pp"] - 2.0) < 1e-9
    assert abs(rows.loc["z_history", "val_ba_interaction_pp"] - 1.0) < 1e-9


def test_training_selection_uses_validation_only() -> None:
    source = inspect.getsource(exp.train_one)
    assert '"val"' in source
    assert "SELECTOR_BA" in source
    assert '"test"' not in source
    assert "test_ba" not in source


def test_finalizer_hard_fails_on_multiple_hosts() -> None:
    source = inspect.getsource(exp.finalize)
    assert "if len(hosts) != 1" in source
    assert "requires one physical node" in source
    assert "single_node_verified" in source


def test_launcher_runs_all_tasks_inside_current_allocation() -> None:
    source = inspect.getsource(exp.launch)
    assert "ThreadPoolExecutor" in source
    assert "max_workers=max_workers" in source
    assert "range(count)" in source
    assert "subprocess" in inspect.getsource(exp._launch_one)


def test_slurm_script_requests_one_node_thirty_cpus() -> None:
    root = exp.find_repo_root()
    script = (
        root
        / "scripts"
        / "bash_script"
        / "SNN_Bash"
        / "run_exp_16_1_single_node_cpu.bash"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-task=30" in script
    assert "#SBATCH --mem=100G" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "launch-train" in script
    assert "launch-ablation" in script
    assert "OMP_NUM_THREADS=1" in script


def test_functional_ablation_contract() -> None:
    assert exp.FUNCTIONAL_ABLATIONS == (
        "learned",
        "gate_one",
        "gate_time_mean",
        "gate_time_shuffle",
    )
    assert exp.SHUFFLE_SEEDS == (101, 211, 307)

