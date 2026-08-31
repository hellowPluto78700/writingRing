from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import with_gyro_experiment_0_1_temporal_representation_probe as exp01


EXPECTED_CONDA_PREFIX_FRAGMENT = "/work/pi_jgummeso_umass_edu/${USER}/.conda/envs/writingring-gpu"


def test_protocol_dimensions_and_run_mapping_are_stable() -> None:
    assert exp01.SPLIT_SEEDS == (11, 23, 37, 53, 71)
    assert exp01.FIXED_DURATION_MS == (50.0, 150.0)
    assert exp01.RELATIVE_N_BINS == (1, 2, 4, 6, 8, 10, 12, 16, 20)
    assert exp01.EVENT_CHANNEL_COUNT == 60
    assert exp01.TOTAL_CHANNEL_COUNT == 66
    assert exp01.EXPECTED_SAMPLING_RATE_HZ == 64.0
    assert len(exp01.run_specs()) == 55
    assert len({spec.key for spec in exp01.run_specs()}) == 55


def test_integral_fixed_duration_rounding_matches_64_hz_contract() -> None:
    assert int(np.rint(50.0 * exp01.EXPECTED_SAMPLING_RATE_HZ / 1000.0)) == 3
    assert int(np.rint(150.0 * exp01.EXPECTED_SAMPLING_RATE_HZ / 1000.0)) == 10


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
    assert "#SBATCH --array=0-54%50" in script
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
    assert "finalize_exp_0_1_cpu.bash" in script
