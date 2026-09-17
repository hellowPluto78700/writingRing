from __future__ import annotations

from pathlib import Path

import torch

from scripts import experiment_8_1_1_two_layer_mt_factorial as exp811


def test_factorial_matrix_and_locked_a2_contract() -> None:
    specs = exp811.run_specs()
    assert len(specs) == 12
    assert exp811.SEEDS == (11, 23, 37)
    assert exp811.WIDTH == 128
    assert exp811.SHIFTS == (2, 3, 4)
    assert exp811.THRESHOLD_MULTIPLIERS == (0.5, 1.0, 1.5)
    assert exp811.METHOD_ORDER == ("bb", "mb", "bm", "mm")
    assert exp811.METHOD_CONFIG["bb"] == {
        "l1_coding": "binary",
        "l2_coding": "binary",
    }
    assert exp811.METHOD_CONFIG["mb"] == {
        "l1_coding": "hetero3",
        "l2_coding": "binary",
    }
    assert exp811.METHOD_CONFIG["bm"] == {
        "l1_coding": "binary",
        "l2_coding": "hetero3",
    }
    assert exp811.METHOD_CONFIG["mm"] == {
        "l1_coding": "hetero3",
        "l2_coding": "hetero3",
    }


def test_mt_layout_is_width_preserving_and_crosses_threshold_with_tau() -> None:
    assert exp811._bank_widths() == (43, 43, 42)
    groups = exp811._groups("hetero3")
    assert sum(int(group["count"]) for group in groups) == 128
    assert {float(group["threshold_multiplier"]) for group in groups} == {
        0.5,
        1.0,
        1.5,
    }
    for multiplier in exp811.THRESHOLD_MULTIPLIERS:
        assert {
            int(group["shift"])
            for group in groups
            if float(group["threshold_multiplier"]) == multiplier
        } == {2, 3, 4}


def test_mt_thresholds_are_fixed_binary_not_adaptive() -> None:
    lif = exp811._lif("hetero3", beta=0.0)
    thresholds_before = lif.thresholds.detach().clone()
    current = torch.linspace(0.0, 1.0, 128).unsqueeze(0)
    membrane = torch.zeros_like(current)
    spikes, _, _ = lif(current, membrane)
    assert set(torch.unique(spikes).tolist()).issubset({0.0, 1.0})
    assert torch.equal(lif.thresholds, thresholds_before)
    assert not any("threshold" in name for name, _ in lif.named_parameters())


def test_all_factorial_cells_have_identical_parameter_count() -> None:
    counts = []
    for method in exp811.METHOD_ORDER:
        spec = exp811.RunSpec(method, 11)
        model = exp811.Exp811Net(spec, n_classes=12, fs=64.0)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1


def test_factorial_effect_formula() -> None:
    import pandas as pd

    rows = []
    values = {"bb": 0.50, "mb": 0.55, "bm": 0.52, "mm": 0.60}
    for seed in exp811.SEEDS:
        for method in exp811.METHOD_ORDER:
            row = {
                "method": method,
                "seed": seed,
            }
            for metric in exp811.FACTORIAL_METRICS:
                row[metric] = values[method]
            rows.append(row)
    effects = exp811._factorial_effects(pd.DataFrame(rows))
    subset = effects[
        (effects.seed == 11) & (effects.metric == "l2_communication_fixed250_ba")
    ]
    observed = dict(zip(subset.effect, subset.value, strict=True))
    assert abs(observed["l1_mt_main_effect"] - 0.065) < 1e-12
    assert abs(observed["l2_mt_main_effect"] - 0.035) < 1e-12
    assert abs(observed["l1_x_l2_interaction"] - 0.03) < 1e-12


def test_slurm_and_notebook_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    run_bash = (
        root / "scripts/bash_script/SNN_Bash/run_exp_8_1_1_cpu_array.bash"
    ).read_text()
    final_bash = (
        root / "scripts/bash_script/SNN_Bash/finalize_exp_8_1_1_cpu.bash"
    ).read_text()
    submit_bash = (
        root / "scripts/bash_script/SNN_Bash/submit_exp_8_1_1_cpu.bash"
    ).read_text()
    notebook = (
        root / "notebooks/experiment_8_1_1_two_layer_mt_factorial.ipynb"
    ).read_text()
    assert "#SBATCH --array=0-11%12" in run_bash
    assert "#SBATCH --cpus-per-task=1" in run_bash
    assert "OMP_NUM_THREADS=1" in run_bash
    assert " finalize" in final_bash
    assert 'dependency="afterok:${run_job}"' in submit_bash
    assert "factorial_effect_summary.csv" in notebook
    for forbidden in ("import torch", "sbatch", "run_one(", "subprocess"):
        assert forbidden not in notebook
