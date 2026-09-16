from __future__ import annotations

import inspect
from pathlib import Path

from scripts import experiment_7_3_8_lif_gain_sweep as exp738


def test_exp738_protocol_constants() -> None:
    assert exp738.PROTOCOL_VERSION == "lif_gain_sweep_v1"
    assert exp738.SEEDS == (11, 23, 37)
    assert exp738.GAIN_GRID == (
        0.125,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
    )
    assert exp738.WEIGHT_SOURCES == ("affine_w", "no_bias_w")
    assert [spec.seed for spec in exp738.run_specs()] == [11, 23, 37]


def test_gain_selection_uses_validation_ba_and_smallest_tie() -> None:
    rows = [
        {"weight_source": "affine_w", "split": "val", "gain": 0.5, "balanced_accuracy": 0.4},
        {"weight_source": "affine_w", "split": "val", "gain": 1.0, "balanced_accuracy": 0.5},
        {"weight_source": "affine_w", "split": "val", "gain": 2.0, "balanced_accuracy": 0.5},
        {"weight_source": "affine_w", "split": "test", "gain": 64.0, "balanced_accuracy": 1.0},
    ]
    assert exp738._select_gain(rows, "affine_w") == 1.0


def test_hybrid_control_and_lif_transfer_contract_are_explicit() -> None:
    source = inspect.getsource(exp738)
    assert '"A2_linear_hybrid_W0_b1"' in source
    assert 'no_bias["raw_weight"], affine["raw_bias"]' in source
    assert '"lif_input_bias": "none for both weight sources"' in source
    assert '"lif_weight_training": "none; only a positive scalar current gain multiplies frozen Linear W"' in source
    assert "gain * raw_weight" in source


def test_slurm_array_and_afterok_finalizer() -> None:
    root = Path(__file__).resolve().parents[1]
    run = (root / "scripts/bash_script/SNN_Bash/run_exp_7_3_8_cpu_array.bash").read_text()
    submit = (root / "scripts/bash_script/SNN_Bash/submit_exp_7_3_8_cpu.bash").read_text()
    assert "#SBATCH --array=0-2%3" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "--dependency=\"afterok:${train_job}\"" in submit
