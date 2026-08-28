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


def test_bash_entrypoint_declares_one_user_per_cpu_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "bash_script"
        / "preprocessing_pipeline"
        / "build_angular_accel_66ch_variant.bash"
    ).read_text(encoding="utf-8")

    assert "--array=" in script
    assert "--cpus-per-task=1" in script
    assert "SLURM_MAX_CONCURRENCY" in script
    assert "OMP_NUM_THREADS=1" in script
    assert "MKL_NUM_THREADS=1" in script
    assert "OPENBLAS_NUM_THREADS=1" in script
    assert "NUMEXPR_NUM_THREADS=1" in script


def test_bash_entrypoint_validates_python_and_propagates_it_to_slurm() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "bash_script"
        / "preprocessing_pipeline"
        / "build_angular_accel_66ch_variant.bash"
    ).read_text(encoding="utf-8")

    assert 'candidate="${CONDA_PREFIX}/bin/python"' in script
    assert '"$candidate" "$PYTHON_HELPER" --help' in script
    assert "PYTHON_BIN=${PYTHON_BIN},MODE=worker" in script
    assert "PYTHON_BIN=${PYTHON_BIN},MODE=finalize" in script
    assert 'if ! USERS_OUTPUT="$(list_users)"' in script
    assert "user discovery failed with Python interpreter" in script


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
