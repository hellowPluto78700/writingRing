from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import with_gyro_experiment_0_1_temporal_representation_probe as exp01


EXPECTED_CONDA_PREFIX_FRAGMENT = "/work/pi_jgummeso_umass_edu/${USER}/.conda/envs/writingring-gpu"
EXPECTED_FIXED_DURATIONS = (
    50.0,
    150.0,
    250.0,
    350.0,
    450.0,
    550.0,
    650.0,
    750.0,
    850.0,
    950.0,
    1050.0,
)


def test_protocol_dimensions_and_run_mapping_are_stable() -> None:
    assert exp01.PROTOCOL_VERSION == "linear_angular_accel_channel_ablation_v3"
    assert exp01.SPLIT_SEEDS == (11, 23, 37, 53, 71)
    assert exp01.FIXED_DURATION_MS == EXPECTED_FIXED_DURATIONS
    assert exp01.RELATIVE_N_BINS == (1, 2, 4, 6, 8, 10, 12, 16, 20)
    assert exp01.CHANNEL_SETS == {
        "accel30": (0, 30),
        "angular30": (30, 60),
        "combined60": (0, 60),
    }
    assert exp01.EVENT_CHANNEL_COUNT == 60
    assert exp01.TOTAL_CHANNEL_COUNT == 66
    assert exp01.EXPECTED_SAMPLING_RATE_HZ == 64.0
    assert len(exp01.run_specs()) == 100
    assert len({spec.key for spec in exp01.run_specs()}) == 100
    assert len(exp01.CHANNEL_SETS) * len(exp01.CLASSIFIERS) == 6
    assert len(exp01.run_specs()) * len(exp01.CHANNEL_SETS) * len(exp01.CLASSIFIERS) * 2 == 1200


def test_fixed_duration_rounding_matches_64_hz_contract() -> None:
    samples = [
        int(np.rint(duration * exp01.EXPECTED_SAMPLING_RATE_HZ / 1000.0))
        for duration in EXPECTED_FIXED_DURATIONS
    ]
    assert samples == [3, 10, 16, 22, 29, 35, 42, 48, 54, 61, 67]


def test_channel_slices_are_nonoverlapping_branches_plus_combination() -> None:
    assert exp01._channel_bounds("accel30") == (0, 30)
    assert exp01._channel_bounds("angular30") == (30, 60)
    assert exp01._channel_bounds("combined60") == (0, 60)
    with pytest.raises(ValueError, match="Unknown channel set"):
        exp01._channel_bounds("gyro30")


def test_paired_gain_is_computed_within_split_before_aggregation() -> None:
    rows: list[dict[str, object]] = []
    values = {
        11: {"accel30": 0.50, "angular30": 0.55, "combined60": 0.65},
        23: {"accel30": 0.60, "angular30": 0.50, "combined60": 0.70},
    }
    for split_seed, per_channel in values.items():
        for channel_set, ba in per_channel.items():
            rows.append(
                {
                    "representation_family": "fixed_duration",
                    "condition": "fixed_0250ms",
                    "requested_duration_ms": 250.0,
                    "samples_per_bin": 16,
                    "actual_duration_ms": 250.0,
                    "n_bins": 16,
                    "classifier": "linear",
                    "split_seed": split_seed,
                    "channel_set": channel_set,
                    "balanced_accuracy": ba,
                    "accuracy": ba,
                    "macro_f1": ba,
                }
            )
    paired = exp01._paired_gain_rows(pd.DataFrame(rows)).sort_values("split_seed")
    np.testing.assert_allclose(
        paired["delta_balanced_accuracy_combined60_minus_accel30"],
        [0.15, 0.10],
    )
    np.testing.assert_allclose(
        paired["delta_balanced_accuracy_angular30_minus_accel30"],
        [0.05, -0.10],
    )


def test_user_split_is_disjoint_and_reproducible() -> None:
    users = [f"user_{index}" for index in range(20)]
    frame = pd.DataFrame(
        {
            "user": users,
            "sample_id": [f"{user}/action_0/0" for user in users],
        }
    )
    first = exp01.make_user_split(frame, 11)
    second = exp01.make_user_split(frame, 11)
    for name in ("train", "val", "test"):
        assert first[name].user.tolist() == second[name].user.tolist()
    train = set(first["train"].user)
    val = set(first["val"].user)
    test = set(first["test"].user)
    assert len(train) == 12
    assert len(val) == 4
    assert len(test) == 4
    assert not (train & val or train & test or val & test)


def test_contract_rejects_non_angular66_metadata() -> None:
    payload = {
        "feature_schema": "polarity_split_wavelet_events_plus_imu_v1",
        "event_feature_schema": "custom_wavelet_polarity_split_abs_events_v1",
        "event_representation": "unsigned",
        "event_channel_count": 30,
        "channel_count": 36,
        "sampling_rate_hz": 64.0,
    }
    with pytest.raises(ValueError, match="feature_schema"):
        exp01._validate_contract(payload, context="fixture")


def _read_bash(name: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root / "scripts" / "bash_script" / "withGyro" / name
    ).read_text(encoding="utf-8")


def test_slurm_array_uses_one_core_and_pinned_work_conda_prefix() -> None:
    script = _read_bash("run_exp_0_1_cpu_array.bash")
    assert "#SBATCH --array=0-99%50" in script
    assert "TASK_ID >= 100" in script
    assert "#SBATCH --cpus-per-task=1" in script
    assert "OMP_NUM_THREADS=1" in script
    assert "MKL_NUM_THREADS=1" in script
    assert "OPENBLAS_NUM_THREADS=1" in script
    assert "NUMEXPR_NUM_THREADS=1" in script
    assert EXPECTED_CONDA_PREFIX_FRAGMENT in script
    assert 'conda activate "$WRITINGRING_CONDA_PREFIX"' in script
    assert "conda activate writingring-gpu" not in script
    assert 'python -c "import numpy, pandas, sklearn"' in script


def test_finalizer_uses_same_pinned_work_conda_prefix() -> None:
    script = _read_bash("finalize_exp_0_1_cpu.bash")
    assert EXPECTED_CONDA_PREFIX_FRAGMENT in script
    assert 'conda activate "$WRITINGRING_CONDA_PREFIX"' in script
    assert "conda activate writingring-gpu" not in script
    assert 'python -c "import numpy, pandas, sklearn"' in script


def test_submit_pipeline_preflights_prefix_and_uses_afterok_finalizer() -> None:
    script = _read_bash("submit_exp_0_1_pipeline.bash")
    assert EXPECTED_CONDA_PREFIX_FRAGMENT in script
    assert 'conda activate "$WRITINGRING_CONDA_PREFIX"' in script
    assert "conda activate writingring-gpu" not in script
    assert 'python -c "import numpy, pandas, sklearn"' in script
    assert "with_gyro_experiment_0_1_temporal_representation_probe describe" in script
    assert '--export="ALL,WRITINGRING_CONDA_PREFIX=${WRITINGRING_CONDA_PREFIX}"' in script
    assert '--dependency="afterok:${ARRAY_JOB}"' in script
    assert "100 = 5 split seeds x (11 fixed-duration + 9 relative-progress conditions)" in script
    assert "finalize_exp_0_1_cpu.bash" in script
