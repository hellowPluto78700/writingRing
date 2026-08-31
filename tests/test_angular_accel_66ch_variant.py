from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import build_angular_accel_66ch_variant as angular66


def test_angular_acceleration_central_difference_preserves_linear_derivative() -> None:
    rate = 64.0
    time = np.arange(32, dtype=np.float64) / rate
    gyro = np.column_stack((2.0 * time + 1.0, -3.0 * time, 0.5 * time - 4.0))

    result = angular66.angular_acceleration_from_gyro(
        gyro,
        sampling_rate_hz=rate,
    )

    assert result.shape == gyro.shape
    np.testing.assert_allclose(
        result,
        np.tile(np.array([2.0, -3.0, 0.5]), (len(time), 1)),
        rtol=0.0,
        atol=1e-12,
    )


def test_combined_layout_preserves_source_event_and_trailing_imu_channels() -> None:
    rng = np.random.default_rng(17)
    source = np.zeros((11, 36), dtype=np.float32)
    source[:, :30] = rng.random((11, 30), dtype=np.float32)
    source[:, 30:] = rng.normal(size=(11, 6)).astype(np.float32)
    angular = rng.random((11, 30), dtype=np.float32)

    combined = angular66.combine_event_branches(source, angular)

    assert combined.shape == (11, 66)
    assert combined.dtype == source.dtype
    np.testing.assert_array_equal(combined[:, :30], source[:, :30])
    np.testing.assert_array_equal(combined[:, 30:60], angular)
    np.testing.assert_array_equal(combined[:, 60:], source[:, 30:])


def test_angular_encoder_uses_planned_64_hz_wavelet_contract() -> None:
    encoder = angular66.angular_encoder(
        "float32",
    )

    assert encoder.wavelet_widths_samples == (128, 64, 32, 16, 8)
    assert encoder.max_filter_time_samples == 19
    assert encoder.settings.post_encode_transform == "PolaritySplitAbs"
    assert len(encoder.output_channel_names) == 30


def _bash_entrypoint() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root
        / "scripts"
        / "bash_script"
        / "preprocessing_pipeline"
        / "build_angular_accel_66ch_variant.bash"
    ).read_text(encoding="utf-8")


def test_bash_entrypoint_declares_one_user_per_cpu_contract() -> None:
    script = _bash_entrypoint()

    assert "--array=" in script
    assert "--cpus-per-task=1" in script
    assert "SLURM_MAX_CONCURRENCY" in script
    assert "OMP_NUM_THREADS=1" in script
    assert "MKL_NUM_THREADS=1" in script
    assert "OPENBLAS_NUM_THREADS=1" in script
    assert "NUMEXPR_NUM_THREADS=1" in script


def test_bash_entrypoint_auto_activates_writingring_gpu_like_exp34() -> None:
    script = _bash_entrypoint()

    assert "module load conda/latest" in script
    assert 'eval "$(conda shell.bash hook)"' in script
    assert "conda activate writingring-gpu" in script
    assert 'PYTHON_CMD=(python)' in script
    assert 'python "$PYTHON_HELPER" --help' in script
    assert 'if ! USERS_OUTPUT="$(list_users)"' in script
    assert "user discovery failed in writingring-gpu" in script
    assert "PYTHON_BIN=" not in script


def test_bash_entrypoint_propagates_canonical_repo_root_to_slurm_jobs() -> None:
    script = _bash_entrypoint()

    assert 'EXPLICIT_REPO_ROOT="${WRITINGRING_REPO_ROOT:-}"' in script
    assert 'export WRITINGRING_REPO_ROOT="$REPO_ROOT"' in script
    assert "resolve_repo_root()" in script
    assert 'WRITINGRING_REPO_ROOT=${REPO_ROOT}' in script
    assert '--chdir="$REPO_ROOT"' in script
    assert 'SCRIPT_PATH="${REPO_ROOT}/scripts/bash_script/preprocessing_pipeline/build_angular_accel_66ch_variant.bash"' in script
    assert 'PYTHON_HELPER="${REPO_ROOT}/scripts/build_angular_accel_66ch_variant.py"' in script
    assert 'REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"' not in script


def test_output_schema_and_channel_counts_are_stable() -> None:
    assert angular66.OUTPUT_SCHEMA == (
        "linear_accel_angular_accel_polarity_split_wavelet_events_plus_imu_v1"
    )
    assert angular66.OUTPUT_EVENT_SCHEMA == (
        "linear_accel_angular_accel_polarity_split_abs_events_v1"
    )
    assert angular66.SOURCE_CHANNELS == 36
    assert angular66.OUTPUT_EVENTS == 60
    assert angular66.OUTPUT_CHANNELS == 66
