from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import experiment_0_1_growing_prefix_relative10 as exp01


def test_exp01_protocol_and_run_matrix() -> None:
    assert exp01.EXPERIMENT_ID == "experiment_0_1_growing_prefix_relative10"
    assert exp01.PROTOCOL_VERSION == "frozen_full_linear_v1"
    assert exp01.SPLIT_SEEDS == (11, 23, 37, 53, 71)
    assert exp01.PREFIX_STEP_SAMPLES == 10
    assert exp01.RELATIVE_N_BINS == 10
    specs = exp01.run_specs()
    assert len(specs) == 5
    assert exp01.EXPECTED_RUNS == 5
    assert [spec.split_seed for spec in specs] == [11, 23, 37, 53, 71]


def test_exp01_relative10_prefix_rebins_entire_observed_prefix() -> None:
    x10 = np.arange(10 * 30, dtype=np.float32).reshape(10, 30)
    f10 = exp01.relative10_from_sequence(x10)
    assert f10.shape == (300,)
    assert np.array_equal(f10.reshape(10, 30), x10)

    x20 = np.ones((20, 30), dtype=np.float32)
    f20 = exp01.relative10_from_sequence(x20).reshape(10, 30)
    assert np.all(f20 == 2.0)

    x30 = np.ones((30, 30), dtype=np.float32)
    f30 = exp01.relative10_from_sequence(x30).reshape(10, 30)
    assert np.all(f30 == 3.0)


def test_exp01_relative10_rejects_prefix_shorter_than_ten_samples() -> None:
    x = np.zeros((9, 30), dtype=np.float32)
    try:
        exp01.relative10_from_sequence(x)
    except ValueError as error:
        assert "at least 10 observed samples" in str(error)
    else:
        raise AssertionError("Expected short prefix to be rejected")


def test_exp01_scaler_is_fit_on_full_training_features() -> None:
    train = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float64)
    mean, std = exp01.fit_full_scaler(train)
    assert np.allclose(mean, [2.0, 4.0])
    assert np.allclose(std, [1.0, 2.0])
    z = exp01.apply_scaler(train, mean, std)
    assert np.allclose(z.mean(axis=0), 0.0)
    assert np.allclose(z.std(axis=0), 1.0)


def test_exp01_ticks_advance_by_ten_samples() -> None:
    import pandas as pd

    frame = pd.DataFrame({"valid_length": [17, 38, 64]})
    assert exp01.growing_prefix_ticks(frame) == [10, 20, 30, 40, 50, 60]


def test_exp01_slurm_array_is_one_split_per_cpu() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_0_1_cpu_array.bash"
    ).read_text(encoding="utf-8")
    finalizer = (
        repo_root / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_0_1_cpu.bash"
    ).read_text(encoding="utf-8")
    submitter = (
        repo_root / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_0_1_pipeline.bash"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --array=0-4%5" in runner
    assert "#SBATCH --cpus-per-task=1" in runner
    assert "OMP_NUM_THREADS=1" in runner
    assert "MKL_NUM_THREADS=1" in runner
    assert "OPENBLAS_NUM_THREADS=1" in runner
    assert "NUMEXPR_NUM_THREADS=1" in runner
    assert (
        "python -u -m scripts.experiment_0_1_growing_prefix_relative10 "
        "run-one --array-task-id \"$TASK_ID\""
    ) in runner
    assert "python -u -m scripts.experiment_0_1_growing_prefix_relative10 finalize" in finalizer
    assert "--dependency=afterok:${EXP01_ARRAY}" in submitter
